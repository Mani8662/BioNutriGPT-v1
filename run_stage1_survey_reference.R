library(survey)
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1], winslash='/')
out <- file.path(root,'Saved Model/stage1_survey_reference_v1')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
writeLines(c('Frozen calibrated predictions; no fitting or threshold selection.',
 'WTDRD1PP; stratified JKn, mse=TRUE; t intervals with design degrees of freedom.',
 'Domain estimates preserve the full replicate design; subgroup outcomes are exploratory.',
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
  d[[paste(model,target,sep='__')]] <- f[[paste0(target,'_calibrated')]]
 }
}
design <- svydesign(ids=~SDMVPSU,strata=~SDMVSTRA,weights=~WTDRD1PP,data=d,nest=TRUE)
repdesign <- as.svrepdesign(design,type='JKn',mse=TRUE)
df <- degf(repdesign)
writeLines(c(paste('R',getRversion()),paste('survey',packageVersion('survey')),paste('df',df)),file.path(out,'environment.txt'))
groups <- list(Overall=rep(TRUE,nrow(d)),Female=d$RIAGENDR==2,Male=d$RIAGENDR==1,
 Age20_40=d$RIDAGEYR<=40,Age41_65=d$RIDAGEYR>=41 & d$RIDAGEYR<=65,AgeOver65=d$RIDAGEYR>65)
for(code in 1:5) groups[[paste0('RIDRETH1_',code)]] <- d$RIDRETH1==code
metric <- function(y,p,w){
 keep <- w>0; y<-y[keep];p<-p[keep];w<-w[keep]
 if(!length(w)||sum(w*y)==0||sum(w*(1-y))==0) return(c(auroc=NA,brier=if(length(w))weighted.mean((y-p)^2,w) else NA,average_precision=NA))
 ord<-order(p); ps<-p[ord]; ids<-cumsum(c(TRUE,diff(ps)!=0)); mat<-rowsum(cbind(pos=w[ord]*y[ord],neg=w[ord]*(1-y[ord])),ids,reorder=FALSE); g<-as.data.frame(mat)
 auc<-sum(g$pos*(cumsum(g$neg)-g$neg/2))/(sum(g$pos)*sum(g$neg))
 g<-g[nrow(g):1,];ap<-sum(g$pos/sum(g$pos)*cumsum(g$pos)/cumsum(g$pos+g$neg))
 c(auroc=auc,brier=weighted.mean((y-p)^2,w),average_precision=ap)
}
rows<-list();pairs<-list()
for(group in names(groups)){
 mask<-groups[[group]]
 for(target in targets){
  for(model in models){
   f<-function(w,data)metric(data[[target]],data[[paste(model,target,sep='__')]],w*mask)
   fit<-withReplicates(repdesign,f)
   value<-coef(fit);se<-SE(fit)
   rows[[length(rows)+1]]<-data.frame(group=group,model=model,outcome=target,metric=names(value),estimate=as.numeric(value),se=as.numeric(se),lower=value-qt(.975,df)*se,upper=value+qt(.975,df)*se,n=sum(mask),positives=sum(d[[target]][mask]),df=df)
   if(model!='bionutrigpt'){
    pf<-function(w,data)metric(data[[target]],data[[paste('bionutrigpt',target,sep='__')]],w*mask)-metric(data[[target]],data[[paste(model,target,sep='__')]],w*mask)
    fitp<-withReplicates(repdesign,pf);v<-coef(fitp);s<-SE(fitp)
    pairs[[length(pairs)+1]]<-data.frame(group=group,comparator=model,outcome=target,metric=names(v),difference=as.numeric(v),se=as.numeric(s),lower=v-qt(.975,df)*s,upper=v+qt(.975,df)*s)
   }
  }
 }
 write.csv(do.call(rbind,rows),file.path(out,'metrics.csv'),row.names=FALSE)
 write.csv(do.call(rbind,pairs),file.path(out,'paired_differences.csv'),row.names=FALSE)
 cat('Completed',group,'\n');flush.console()
}
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
writeLines('R evaluation finished; Python cross-check required before completion claim.',file.path(out,'run_finished.txt'))

