library(survey)
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1], winslash='/')
out <- file.path(root,'Saved Model/stage1_calibration_dca_v1')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
writeLines(c('Frozen calibrated predictions; no fitting or threshold selection.',
 'WTDRD1PP; stratified JKn, mse=TRUE; t intervals with design degrees of freedom.',
 'Calibration diagnostics only: offset intercept and joint intercept/slope, raw and frozen calibrated probabilities; epsilon 1e-7. No recalibration deployment.',
 'Fixed probability bins [0,.1),...,[.9,1]; descriptive observed versus predicted; bin N<100 or events/non-events<20 flagged.',
 'DCA calibrated diabetes/CVD only: thresholds .05 to .50 step .01, hypothetical further assessment, not treatment or nutrition benefit. Existing diagnoses are included; no undiagnosed-only utility claim.',
 'Paired net-benefit contrasts versus all, none and every competing model use shared JKn replicates; pointwise unadjusted intervals. No threshold selected from temporal outcomes.',
 'Complete-case selection is not corrected by survey weights; no national or Indian validation claim.'),file.path(out,'protocol.txt'))
meta <- read.csv(file.path(root,'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv'))
truth <- read.csv(file.path(root,'Data/stage1_repaired_v2/temporal_ml_ready.csv'))
truth <- truth[order(truth$SEQN),]
d <- meta[match(truth$SEQN,meta$SEQN),]
stopifnot(nrow(d)==7004,!anyDuplicated(d$SEQN),all(d$WTDRD1PP>0))
models <- c('logistic_regression','hist_gradient_boosting','catboost','mlp','bionutrigpt','ft_transformer')
targets <- c('TARGET_DIABETES','TARGET_CVD','TARGET_OBESITY')
for (target in targets) d[[target]] <- truth[[target]]
for(model in models){
 f <- read.csv(file.path(root,'Saved Model/stage1_comparison_v2',model,'temporal_predictions_LOCAL.csv'))
 f <- f[match(d$SEQN,f$SEQN),]
 stopifnot(all(f$SEQN==d$SEQN))
 for(target in targets){
  stopifnot(all(f[[target]]==d[[target]]))
  for(variant in c('raw','calibrated')) d[[paste(model,target,variant,sep='__')]] <- f[[paste0(target,'_',variant)]]
 }
}
design <- svydesign(ids=~SDMVPSU,strata=~SDMVSTRA,weights=~WTDRD1PP,data=d,nest=TRUE)
repdesign <- as.svrepdesign(design,type='JKn',mse=TRUE)
df <- degf(repdesign)
writeLines(c(paste('R',getRversion()),paste('survey',packageVersion('survey')),paste('df',df)),file.path(out,'environment.txt'))

calrows<-list();bins<-list();dcrows<-list();pairrows<-list()
for(target in targets){
 for(model in models){
  for(variant in c('raw','calibrated')){
   p<-d[[paste(model,target,variant,sep='__')]]
   stopifnot(all(is.finite(p)),all(p>=0 & p<=1))
   dd<-repdesign;dd$variables$y<-d[[target]];dd$variables$z<-qlogis(pmin(pmax(p,1e-7),1-1e-7))
   fits<-list(offset=svyglm(y~1+offset(z),design=dd,family=quasibinomial()),joint=svyglm(y~z,design=dd,family=quasibinomial()))
   for(kind in names(fits)){
    fit<-fits[[kind]];stopifnot(fit$converged);v<-coef(fit);se<-SE(fit)
    for(k in seq_along(v)) calrows[[length(calrows)+1]]<-data.frame(model=model,outcome=target,variant=variant,fit=kind,term=names(v)[k],estimate=v[k],se=se[k],lower=v[k]-qt(.975,df)*se[k],upper=v[k]+qt(.975,df)*se[k])
   }
   for(b in 0:9){
    mask<-p>=b/10 & (p<(b+1)/10 | (b==9 & p<=1));n<-sum(mask);pos<-sum(d[[target]][mask])
    bins[[length(bins)+1]]<-data.frame(model=model,outcome=target,variant=variant,bin=b,n=n,positive=pos,mean_predicted=if(n)weighted.mean(p[mask],d$WTDRD1PP[mask]) else NA,observed=if(n)weighted.mean(d[[target]][mask],d$WTDRD1PP[mask]) else NA,small=n<100 || pos<20 || n-pos<20)
   }
  }
 }
 if(target!='TARGET_OBESITY'){
  thresholds<-seq(.05,.50,.01)
  # Net benefit is a survey-weighted mean of a fixed participant contribution.
  nb<-function(w,data){
   y<-data[[target]];v<-c()
   for(th in thresholds){
    harm<-th/(1-th)
    for(model in c(models,'all','none')){
     chosen<-if(model=='all')rep(TRUE,length(y)) else if(model=='none')rep(FALSE,length(y)) else data[[paste(model,target,'calibrated',sep='__')]]>=th
     v[paste(model,sprintf('%.2f',th),sep='__')]<-weighted.mean(chosen*(y-(1-y)*harm),w)
    }
   }
   v
  }
  fit<-withReplicates(repdesign,nb);v<-coef(fit);cov<-vcov(fit);se<-sqrt(diag(cov))
  for(k in seq_along(v)){
   label<-strsplit(names(v)[k],'__',fixed=TRUE)[[1]]
   dcrows[[length(dcrows)+1]]<-data.frame(outcome=target,model=label[1],threshold=as.numeric(label[2]),net_benefit=v[k],se=se[k],lower=v[k]-qt(.975,df)*se[k],upper=v[k]+qt(.975,df)*se[k])
  }
  for(th in thresholds){
   bi<-match(paste('bionutrigpt',sprintf('%.2f',th),sep='__'),names(v))
   for(comp in c(setdiff(models,'bionutrigpt'),'all','none')){
    ci<-match(paste(comp,sprintf('%.2f',th),sep='__'),names(v));delta<-v[bi]-v[ci];ss<-sqrt(max(0,cov[bi,bi]+cov[ci,ci]-2*cov[bi,ci]))
    pairrows[[length(pairrows)+1]]<-data.frame(outcome=target,comparator=comp,threshold=th,difference=delta,se=ss,lower=delta-qt(.975,df)*ss,upper=delta+qt(.975,df)*ss)
   }
  }
 }
 write.csv(do.call(rbind,calrows),file.path(out,'calibration.csv'),row.names=FALSE)
 write.csv(do.call(rbind,bins),file.path(out,'calibration_bins.csv'),row.names=FALSE)
 if(length(dcrows))write.csv(do.call(rbind,dcrows),file.path(out,'decision_curves.csv'),row.names=FALSE)
 if(length(pairrows))write.csv(do.call(rbind,pairrows),file.path(out,'decision_curve_pairs.csv'),row.names=FALSE)
 cat('Completed',target,'\n');flush.console()
}
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
writeLines('Awaiting independent verification',file.path(out,'run_finished.txt'))
