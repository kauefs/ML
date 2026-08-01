#@title FastAPI
import joblib
import pandas              as   pd
from   pathlib           import Path
from   pydantic          import BaseModel
from   fastapi           import FastAPI, HTTPException
from   fastapi.responses import RedirectResponse
# Initialize FastAPI App
app=FastAPI(title='Credit Risk Scoring API', description='Live inference service for default probability & credit decisions.', version='1.0.0')
# ---------------------------------------------------------
# Global State & Model Loading
# ---------------------------------------------------------
PATH    =Path('CreditRiskPipeLine.joblib')
artifact=None
@app.on_event('startup')
def loadArtiFact( ):
    global artifact
    if not PATH.exists( ):raise FileNotFoundError(f'ArtiFact Not Found @ {PATH}')
    artifact=joblib.load(PATH)
    print(f'Loaded PipeLine & MetaData from {PATH}')
# ---------------------------------------------------------
# Input Schema Definition (Pydantic Model for API Requests)
# ---------------------------------------------------------
class CreditApplication(BaseModel):
    credit_limit:   float
    sex:              int
    education:        int
    marital_status:   int
    age:              int
    pay_status_sep:   int
    pay_status_aug:   int
    pay_status_jul:   int
    pay_status_jun:   int
    pay_status_may:   int
    pay_status_apr:   int
    bill_amount_sep:float
    bill_amount_aug:float
    bill_amount_jul:float
    bill_amount_jun:float
    bill_amount_may:float
    bill_amount_apr:float
    payment_sep:    float
    payment_aug:    float
    payment_jul:    float
    payment_jun:    float
    payment_may:    float
    payment_apr:    float
    class Config:
        # Sample PayLoad for InterActive OpenAPI Docs
        schema_extra={'example':{'credit_limit'   :20000.,'sex':2,'education':     2,'marital_status' :    1,'age':24,
                                 'pay_status_sep' :     2,'pay_status_aug'   :     2,'pay_status_jul' :   -1,
                                 'pay_status_jun' :    -1,'pay_status_may'   :    -2,'pay_status_apr' :   -2,
                                 'bill_amount_sep':3913.0,'bill_amount_aug'  :3102.0,'bill_amount_jul':689.0,
                                 'bill_amount_jun':    .0,'bill_amount_may'  :    .0,'bill_amount_apr':   .0,
                                 'payment_sep'    :    .0,'payment_aug'      : 689.0,'payment_jul'    :   .0,
                                 'payment_jun'    :    .0,'payment_may'      :    .0,'payment_apr'    :   .0}}
# ---------------------------------------------------------
# EndPoints
# ---------------------------------------------------------
@app.get('/')
def root(   ):
    '''Redirects root URL to interactive Swagger API documentation.'''
    return RedirectResponse(url='/docs')
@app.get('/health')
def healthCheck( ):
    '''Health Check EndPoint for Container Orchestrators.'''
    return{  'status':'healthy','model_loaded': artifact is not None,
           'versions':artifact.get('versions',{ })if artifact else{ }}
@app.post('/predict')
def predictCreditRisk(application: CreditApplication):
    '''Generates default risk probability and operational decision (Approve/Deny).'''
    try:
        # 1. Convert request payload into DataFrame
        input_data   = pd.DataFrame([application.dict()])
        # 2. Enforce schema & feature alignment
        expected_cols= artifact['feature_names']
        missing_cols = set(expected_cols)-set(input_data.columns)
        if missing_cols:raise HTTPException (status_code=400, detail=f'Missing feature columns: {missing_cols}')
        prepared_data= input_data.loc[:,   expected_cols]
        # 3. Inference pass through fitted pipeline
        prob=float(artifact['pipeline'].predict_proba(prepared_data)[:,1][0])
        threshold =artifact['threshold']
        pred=int(prob   >=   threshold)
        # 4. Construct response payload
        return{'prob_default'  : round(    prob , 4),
               'threshold_used': round(threshold, 4),
               'prediction'    : pred,
               'label'         :'DeFault'      if pred==1 else'NonDeFault',
               'action'        :'Deny/HighRisk'if pred==1 else'Approve'}
    except Exception as e:raise HTTPException(status_code=500, detail=str(e))
