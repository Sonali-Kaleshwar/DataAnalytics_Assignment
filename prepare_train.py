"""Train reproducible baseline and ML models on ASSISTments corrected data.
Run: python prepare_train.py
"""
import json, zipfile
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,brier_score_loss
ROOT=Path(__file__).resolve().parent
(ROOT/'models').mkdir(exist_ok=True)
(ROOT/'outputs').mkdir(exist_ok=True)
source=ROOT/'data'/'source.zip'
with zipfile.ZipFile(source) as z:
    with z.open(next(n for n in z.namelist() if n.endswith('.csv'))) as f:
        df=pd.read_csv(f,encoding='latin1',low_memory=False)
summary={'original_rows':int(len(df)), 'column_names':df.columns.tolist(), 'missing_skill_name':int(df.skill_name.isna().sum()),'unique_students':int(df.user_id.nunique()),'unique_questions':int(df.problem_id.nunique())}
df=df.dropna(subset=['user_id','order_id','skill_name','correct']).copy()
df=df[df['correct'].isin([0,1])]
df['skill_name']=df['skill_name'].astype(str).str.strip()
df=df[df.skill_name.ne('')]
df=df.sort_values(['order_id','user_id'],kind='stable').reset_index(drop=True)
# Historical student and skill features, strictly excluding the current target.
def historical(group_cols, prefix):
    g=df.groupby(group_cols,sort=False)['correct']
    n=g.cumcount().astype('float64')
    total=g.cumsum().astype('float64')-df['correct'].to_numpy()
    df[prefix+'_prior_n']=n
    df[prefix+'_prior_rate']=np.divide(total,n,out=np.full(len(df),0.5),where=n.to_numpy()>0)
historical(['user_id'],'student')
historical(['skill_name'],'skill')
historical(['user_id','skill_name'],'student_skill')
# Numeric/encoded features only known prior to current question response.
NUM=['student_prior_n','student_prior_rate','skill_prior_n','skill_prior_rate','student_skill_prior_n','student_skill_prior_rate']
CAT=['skill_name']
FEATURES=NUM+CAT
# Treat final 20% chronological interactions as the held-out evaluation set.
cut=int(len(df)*0.8)
train,test=df.iloc[:cut].copy(),df.iloc[cut:].copy()
Xtr,Xte=train[FEATURES],test[FEATURES]
ytr,yte=train['correct'].astype(int),test['correct'].astype(int)
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('skill',OneHotEncoder(handle_unknown='ignore'),CAT)])
models={
    'Logistic Regression':Pipeline([('pre',pre),('classifier',LogisticRegression(max_iter=400,random_state=42))]),
    'Decision Tree':Pipeline([('pre',pre),('classifier',DecisionTreeClassifier(max_depth=6,min_samples_leaf=150,random_state=42))])
}
def evaluate(yp,p):
    return dict(accuracy=round(float(accuracy_score(yte,yp)),4),precision=round(float(precision_score(yte,yp,zero_division=0)),4),recall=round(float(recall_score(yte,yp,zero_division=0)),4),f1=round(float(f1_score(yte,yp,zero_division=0)),4),roc_auc=round(float(roc_auc_score(yte,p)),4),brier=round(float(brier_score_loss(yte,p)),4),confusion_matrix=confusion_matrix(yte,yp).tolist())
results={}
base_p=float(ytr.mean())
results['Majority baseline']=evaluate(np.full(len(yte),int(base_p>=0.5)),np.full(len(yte),base_p))
for name,model in models.items():
    print('Training',name,flush=True)
    model.fit(Xtr,ytr)
    probs=model.predict_proba(Xte)[:,1]
    results[name]=evaluate((probs>=.5).astype(int),probs)
    joblib.dump(model,ROOT/'models'/('logistic_regression.joblib' if name=='Logistic Regression' else 'decision_tree.joblib'))
summary.update({'retained_rows':int(len(df)),'unique_skills':int(df.skill_name.nunique()),'training_rows':int(len(train)),'test_rows':int(len(test)),'train_correct_rate':round(base_p,4),'test_correct_rate':round(float(yte.mean()),4),'train_max_order_id':str(train.order_id.max()),'test_min_order_id':str(test.order_id.min()),'model_features':FEATURES,'metrics':results,'notes':['Prediction is first-attempt correctness as labelled by the dataset; no current-attempt hint_count or attempt_count used.','Ordering assumes order_id captures interaction chronology. Validate this if additional timestamps become available.','Historical features for held-out records use earlier observed labels, representing sequential deployment; this is not a cold-start evaluation.','The prediction model is not automatically retrained when new CSV records are imported.']})
(ROOT/'outputs'/'metrics.json').write_text(json.dumps(summary,indent=2))
# compact data for reproducible dashboard; no identifying names
sample=df[['order_id','user_id','problem_id','skill_name','correct']+NUM].copy()
sample.to_csv(ROOT/'data'/'assessment_features.csv',index=False)
print(json.dumps({'rows':summary['retained_rows'],'test':summary['test_rows'],'scores':results},indent=2),flush=True)
