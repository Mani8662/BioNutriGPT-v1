"""Retain 30 measured pilot solves; recalculate audits under repaired analysis.
No solver calls. Original v2 records remain unchanged and hash-linked.
"""
import copy
import stage3_robustness_v3 as r

def main():
    r.check_manifest()
    old=r.ROOT/'Saved Model/stage3_robustness_v2/pilot/checkpoints'
    base=r.DIR/'pilot/checkpoints'; base.mkdir(parents=True,exist_ok=True)
    migrated=[]
    for t in r.tasks([0]):
        if not t[5]['solver_required']: continue
        for v in ['a','b']:
            source=r.path_for(old,t,v); record=r.read(source)
            require_same=r.read(r.ROOT/'Saved Model/stage3_robustness_v2/scenario_manifest.json')['scenarios'][t[4]]
            r.require(require_same==t[5],'Scenario changed; cannot reuse solve')
            record['binding']=r.binding(t)
            record['audit']=r.assess(record['basket'],t,r.catalog())
            r.require(record['accepted']==record['audit']['valid'],'Acceptance changed on repair')
            record['derived_from']={'path':str(source.relative_to(r.ROOT)),'sha256':r.sha(source),'reason':'Target-error audit excludes report-only unknowns; measured solve unchanged'}
            r.validate_record(record,t,v,base)
            dest=r.path_for(base,t,v)
            if dest.exists(): r.require(r.read(dest)==record,'Existing migrated record differs')
            else: r.write(dest,record)
            migrated.append(record['derived_from'])
    r.write(r.DIR/'pilot/migration.json',{'solver_calls':0,'reused_measured_solves':len(migrated),'sources':migrated,'script_sha256':r.sha(__file__)})
    print('Migrated measured solver records:',len(migrated))
if __name__=='__main__': main()
