from mangum import Mangum

from lambda_api.app import app

handler = Mangum(app, lifespan="on")
