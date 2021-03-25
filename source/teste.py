#import pickle as pickle
import pandas as pd
import numpy as np
import requests

umaLinha = pd.DataFrame({
    'fLength':[19.2512],
    'fWidth':[14.7951],
    'fSize':[2.2954],
    'fM3Long':[8.9581],
    'fAlpha':[51.6492]
})


import sys
import numpy as np

stack_name = sys.argv[1]
commit_id = sys.argv[2]
endpoint_name = f"{stack_name}-{commit_id[:7]}"

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, Body=umaLinha, ContentType="application/json"
)

result = response["Body"].read()
result = json.loads(result.decode("utf-8"))

print(f"Probabilities: {result}")