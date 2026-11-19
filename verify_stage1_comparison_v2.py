"""Synthetic-only checks: never run the participant benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_stage1_comparison_v2 as runner


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads(runner.PROTOCOL_PATH.read_text())
        self.rng=np.random.default_rng(19)

    def synthetic(self):
        result={};offset=0
        for name,n in [('training',128),('validation',64),('calibration',64),('temporal',64)]:
            x=self.rng.normal(size=(n,87)).astype(np.float32)
            y=np.column_stack([(x[:,i]+self.rng.normal(size=n)>.0).astype(np.float32) for i in range(3)])
            result[name]={'x':x,'y':y,'ids':np.arange(offset,offset+n)};offset+=n
        return result

    def test_schema_reordering_and_overlapping_ids_rejected(self):
        features=[f'x{i}' for i in range(87)]
        def frame(ids):
            f=pd.DataFrame({'SEQN':ids,'NHANES_CYCLE':['synthetic']*2})
            for t in runner.TARGETS:f[t]=[0,1]
            return pd.concat([f,pd.DataFrame(np.zeros((2,87)),columns=features)],axis=1)
        a=frame([1,2]);b=frame([3,4]);runner.validate_partitions({'a':a,'b':b},features)
        with self.assertRaises(ValueError):runner.validate_partitions({'a':a,'b':a},features)
        with self.assertRaises(ValueError):runner.validate_partitions({'a':a[list(a.columns[:-2])+list(a.columns[-2:][::-1])]},features)

    def test_selection_ignores_secondary_obesity(self):
        y=np.array([[0,0,0],[1,1,1],[0,1,0],[1,0,1]])
        p=.1+.8*y
        before=runner.selection_score(y,p);p[:,1]=1-p[:,1]
        self.assertEqual(before,runner.selection_score(y,p))

    def test_logit_calibration_and_no_in_place_mutation(self):
        data=self.synthetic();y=data['calibration']['y'];p=np.clip(.2+.6*self.rng.random(y.shape),.01,.99);before=p.copy()
        c=runner.fit_calibrators(y,p);q=runner.apply_calibrators(c,p)
        np.testing.assert_array_equal(p,before)
        self.assertEqual(q.shape,p.shape);self.assertTrue(np.isfinite(q).all())
        for i,cal in enumerate(c):
            expected=runner.expit(cal.intercept_[0]+cal.coef_[0,0]*runner.logit(p[:,i]))
            np.testing.assert_allclose(q[:,i],expected)

    def test_metrics_threshold_and_bin_counts(self):
        y=np.tile([[0,0,0],[1,1,1]],(20,1));p=.2+.6*y
        rows,curves=runner.metrics(y,p)
        self.assertEqual(len(rows),3)
        for row in rows:self.assertEqual(row['auroc'],1.);self.assertAlmostEqual(row['brier'],.04)
        for t in runner.TARGETS:self.assertEqual(sum(c['n'] for c in curves if c['outcome']==t),40)

    def test_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_text('original')
            runner.write_json(root/'completed.json',{'artifact_hashes':{'a':runner.sha(root/'a')}})
            runner.verify_completion(root,'completed.json');(root/'a').write_text('changed')
            with self.assertRaises(ValueError):runner.verify_completion(root,'completed.json')

    def test_cpu_export_resume_and_calibration_isolation(self):
        data=self.synthetic();p=copy.deepcopy(self.protocol);p['models']['logistic_regression']=[{'C':.1},{'C':1.}]
        p['version']='synthetic-test-only';p['data_version']='synthetic-test-only'
        real_fit=runner.fit_calibrators
        def checked_fit(y,prob):
            np.testing.assert_array_equal(y,data['calibration']['y'])
            return real_fit(y,prob)
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'fit_calibrators',side_effect=checked_fit) as spy:
            runner.run_model('logistic_regression',data,[f'x{i}' for i in range(87)],p,Path(tmp))
            self.assertEqual(spy.call_count,1)
            folder=Path(tmp)/'logistic_regression'
            result=runner.verify_completion(folder,'completed.json')
            self.assertEqual(result['n']['temporal'],64)
            df=pd.read_csv(folder/'temporal_predictions_LOCAL.csv')
            np.testing.assert_array_equal(df.SEQN,data['temporal']['ids'])
            with patch.object(runner,'train_trial',side_effect=AssertionError('Completed trials must not rerun')):
                runner.run_model('logistic_regression',data,[],p,Path(tmp))
            self.assertEqual(spy.call_count,1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_neural_gpu_forward_backward_and_checkpoint(self):
        for name in ['mlp','bionutrigpt','ft_transformer']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                cfg=self.protocol['models'][name][0];model=runner.make_neural(name,cfg).cuda()
                x=torch.randn(4,87,device='cuda');y=torch.randint(0,2,(4,3),device='cuda').float()
                out,aux=runner.forward(name,model,x)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(out,y)+.01*aux
                loss.backward();self.assertTrue(torch.isfinite(loss));self.assertEqual(tuple(out.shape),(4,3))
                model.eval();a=runner.predict(name,model,x.cpu().numpy(),'cuda')
                path=Path(tmp)/'model.pt';runner.save_torch(path,{k:v.cpu() for k,v in model.state_dict().items()})
                loaded=runner.load_trial(name,cfg,Path(tmp),'cuda')
                b=runner.predict(name,loaded,x.cpu().numpy(),'cuda');np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
                del model,loaded;torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_catboost_gpu_available(self):
        from catboost import CatBoostClassifier
        x=self.rng.normal(size=(32,4));y=np.tile([0,1],16)
        m=CatBoostClassifier(iterations=2,depth=2,task_type='GPU',devices='0',allow_writing_files=False,verbose=False)
        m.fit(x,y);self.assertTrue(np.isfinite(m.predict_proba(x)).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_nonlinear_training_loops_on_synthetic_data(self):
        data=self.synthetic();p=copy.deepcopy(self.protocol)
        p['neural']['max_epochs']=1
        for name in ['hist_gradient_boosting','catboost','mlp','bionutrigpt','ft_transformer']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                cfg=copy.deepcopy(p['models'][name][2 if name=='ft_transformer' else 0])
                if name=='hist_gradient_boosting':cfg['max_iter']=2
                if name=='catboost':cfg['iterations']=2
                device='cpu' if name=='hist_gradient_boosting' else 'cuda'
                model,prob,summary=runner.train_trial(name,cfg,data['training'],data['validation'],p,Path(tmp),device)
                self.assertEqual(prob.shape,(64,3));self.assertTrue(np.isfinite(prob).all())
                restored=runner.load_trial(name,cfg,Path(tmp),device)
                np.testing.assert_allclose(prob,runner.predict(name,restored,data['validation']['x'],device),rtol=1e-5,atol=1e-6)
                del model,restored;torch.cuda.empty_cache()


if __name__=='__main__':
    torch.set_num_threads(runner.os.cpu_count() or 1)
    unittest.main(verbosity=2)
