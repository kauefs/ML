#  Settings
# Libraries
import       numpy                   as   np
import      pandas                   as   pd
import     xgboost                   as   xgb
import     sklearn, joblib, torch,        sys
from       sklearn.compose         import ColumnTransformer
from       sklearn.dummy           import DummyClassifier
from       sklearn.impute          import SimpleImputer
from       sklearn.preprocessing   import OneHotEncoder, StandardScaler
from       sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, StratifiedKFold, RandomizedSearchCV
from       sklearn.linear_model    import LogisticRegression
from       sklearn.pipeline        import Pipeline
from       sklearn.metrics         import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay
from       xgboost                 import XGBClassifier
from        urllib.request         import urlretrieve
from       pathlib                 import Path
#   Configs
device=torch.accelerator.current_accelerator( ).type if torch.accelerator.is_available( )else'cpu'
seed             =2
np.random.seed   (seed)
torch.manual_seed(seed)
random_state     =seed
session_id       =seed
epochs           =10
# Data Ingestion
# Data DownLoad
URL ='https://github.com/carlosfab/pos-graduacao-sigmoidal/raw/refs/heads/main/PPG-DS/CS301%20-%20Machine%20Learning%20I/turma-3/data/default_credit_card_clients.csv'
PATH='default_credit_card_clients.csv'
urlretrieve(URL,PATH)
# Data Load
MODEL= Path('CreditRiskPipeLine.joblib')
df=pd .read_csv(PATH)
# ReNaming Features
base   =[ 'credit_limit','sex','education','marital_status','age']
status =[f'pay_status_{m}' for m in['sep','aug','jul','jun','may','apr']]
bill   =[f'bill_amount_{m}'for m in['sep','aug','jul','jun','may','apr']]
paid   =[f'payment_{m}'    for m in['sep','aug','jul','jun','may','apr']]
feature=base+status+bill+paid
df.columns=feature+['default_next_month']
# Train & Test Split
XX=df.drop(columns='default_next_month')
YY=df['default_next_month']
X, x, Y, y = train_test_split(XX, YY, test_size=.2, stratify=YY, random_state=random_state)
# TransFormation PipeLine
categorical =['sex','education','marital_status']
numerical   =[col for col in XX.columns if col not in categorical]
numPipe     =Pipeline([('imputer', SimpleImputer(strategy='median')),
                       ('scaler' , StandardScaler( ))])
catPipe     =Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                       ('onehot' , OneHotEncoder(handle_unknown='ignore'))])
preprocessor=ColumnTransformer([('numeric'    , numPipe, numerical),
                                ('categorical', catPipe, categorical)],
                              verbose_feature_names_out=False)
# BaseLine & XGBoost
dummyPipe   =Pipeline([('preprocess', preprocessor),
                       ('model'     , DummyClassifier(strategy='prior'))])
logistPipe  =Pipeline([('preprocess', preprocessor),
                       ('model'     , LogisticRegression(max_iter=2000, random_state=random_state))])
xgbPipe     =Pipeline([('preprocess', preprocessor),
                       ('model'     , XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                                    tree_method='hist', n_estimators=300, max_depth=3,
                                                    learning_rate=.05, random_state=random_state, n_jobs=-1))])
# Cross-Validation
cv     =  StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
scoring={'pr_auc':'average_precision','roc_auc':'roc_auc','balanced_accuracy':'balanced_accuracy'}
models ={'Dummy':dummyPipe,'Logistic':logistPipe,'XGBoost':xgbPipe}
def evaluateCV(name, estimator):
    scores=cross_validate(estimator, X, Y, cv=cv, scoring=scoring, n_jobs=1)
    return {'model':name,
             'PR-AUCstd'      : scores['test_pr_auc' ]          .std ( ),
             'PR-AUCmean'     : scores['test_pr_auc' ]          .mean( ),
            'ROC-AUCmean'     : scores['test_roc_auc']          .mean( ),
            'BalancedAccuracy': scores['test_balanced_accuracy'].mean( )}
resultsCV=pd.DataFrame([evaluateCV(name, model)for name, model in models.items( )])
# HyperParaMeters Search
params={'model__n_estimators'    :[250, 400, 600],
        'model__max_depth'       :[  2,   3,   4],
        'model__learning_rate'   :[.03, .05, .08],
        'model__min_child_weight':[  1,   5,  10],
        'model__subsample'       :[ .7,  .9,  1.],
        'model__colsample_bytree':[ .7,  .9,  1.]}
search=RandomizedSearchCV(xgbPipe, params, n_iter=8, scoring='average_precision', cv=cv,
                          refit=True, random_state=random_state, n_jobs=-1, return_train_score=False)
search.fit(X,Y)
# Out-of-Fold ThresHold
oofProba=cross_val_predict(search.best_estimator_, X, Y, cv=cv, method='predict_proba', n_jobs=-1)[:,1]
precision,recall,thresholds=precision_recall_curve(Y, oofProba)
f1Values =2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-9)
bestThresHold=thresholds[np.argmax(f1Values)]
# Final Evaluation
finalPipe   = search.best_estimator_
proba       = finalPipe.predict_proba(x)[:,1]
pred        =(proba>=bestThresHold).astype(int)
finalMetrics= pd.Series({# Independent Metrics (Model Ranking Quality)
                         'PR-AUC'         :average_precision_score(y, proba),
                         'ROC-AUC'         :roc_auc_score          (y, proba),
                         #   Dependent Metrics (Operational Decisions)
                         'Precision'       :precision_score        (y, pred ),
                         'ReCall'          :recall_score           (y, pred ),
                         'F1'              :f1_score               (y, pred ),
                         'BalancedAccuracy':balanced_accuracy_score(y, pred )})
# PipeLine Persistence
artifact={'pipeline':finalPipe,'threshold':float(bestThresHold),'feature_names':list(X.columns),
          'target_name':'default_next_month','versions':{'scikit-learn':sklearn.__version__,'xgboost':xgb.__version__,'pandas':pd.__version__},
          'metrics':finalMetrics.to_dict( )}
joblib.dump(artifact, MODEL)
print(MODEL, f'{MODEL.stat( ).st_size/1024:.0f}KB')
# Loading ArtiFact
loadedArtiFact =joblib.load    (MODEL)
loadedPipeLine =loadedArtiFact['pipeline' ]
loadedThresHold=loadedArtiFact['threshold']
print(type(loadedPipeLine).__name__)
print(f'Loaded ThresHold  {loadedThresHold:.4f}')
print(loadedArtiFact['versions'])
# Live Prediction
def RiskPredict(Data, ArtiFact):
    columns=ArtiFact['feature_names']
    missing=sorted(set(columns)-set(Data.columns))
    if missing:raise ValueError(f'Missing Columns: {missing}')
    prepared     =Data.loc[:, columns].copy( )
    probabilities=ArtiFact['pipeline'].predict_proba(prepared)[:,1]
    predictions  =(probabilities >= ArtiFact['threshold']).astype(int)
    labels       =pd.Series(predictions, index=Data.index).map({0:'NonDeFault',1:'DeFault'})
    actions      =pd.Series(predictions, index=Data.index).map({0:'Approve'   ,1:'Deny/HighRisk'})
    return pd.DataFrame({'prob_default':probabilities,'prediction':predictions,'label':labels,'action':actions}, index=Data.index)
