from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import sys
import pickle

flask_app = Flask(__name__)
app = Api(app = flask_app,
 version = "1.0",
 title = "Language Level Predictor",
 description = "Based on Position Of Speech tags, predict text level")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params',
 {'A': fields.Float(required = True,
  description="A",
  help="A cannot be blank"),
 'C': fields.Float(required = True,
  description="C",
  help="C cannot be blank"),
 'D': fields.Float(required = True,
  description="D",
  help="D cannot be blank"),
 'G': fields.Float(required = True,
  description="G",
  help="G cannot be blank"),
 'H': fields.Float(required = True,
  description="H",
  help="H cannot be blank"),
 'I': fields.Float(required = True,
  description="I",
  help="I cannot be blank"),
 'J': fields.Float(required = True,
  description="J",
  help="J cannot be blank"),
 'K': fields.Float(required = True,
  description="K",
  help="K cannot be blank"),
 'N': fields.Float(required = True,
  description="N",
  help="N cannot be blank"),
 'O': fields.Float(required = True,
  description="O",
  help="O cannot be blank"),
 'P': fields.Float(required = True,
  description="P",
  help="P cannot be blank"),
 'S': fields.Float(required = True,
  description="S",
  help="S cannot be blank"),
 'U': fields.Float(required = True,
  description="U",
  help="U cannot be blank"),
 'V': fields.Float(required = True,
  description="V",
  help="V cannot be blank"),
 'X': fields.Float(required = True,
  description="X",
  help="X cannot be blank"),
 'Y': fields.Float(required = True,
  description="Y",
  help="Y cannot be blank"),
 'Z': fields.Float(required = True,
  description="Z",
  help="Z cannot be blank"),
  'Z': fields.Float(required = True,
  description="Z",
  help="Z cannot be blank")})


classifier = pickle.load(open('dtree.sav', 'rb'))


@name_space.route("/")
class MainClass(Resource):

  def options(self):
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response
  @app.expect(model)
  def post(self):
    try:
      formData = request.json
      data = [float(val) for val in formData.values()]
      print("predicting ...")
      print('data: ', data)
      prediction = classifier.predict(np.array(data).reshape(1, -1))
      print("predicted")
      print("prediction ", prediction)
      types = { 0: "A2", 1:'B1', 2:'B2', 3:'C1' }
      response = jsonify({
        "statusCode": 200,
        "status": "Prediction made",
        "result": "The language level is: " + types[prediction[0]]
        })
      response.headers.add('Access-Control-Allow-Origin', '*')
      return response
    except Exception as error:
      return jsonify({
        "statusCode": 500,
        "status": "Could not make prediction",
        "error": str(error)
        })


if __name__ == '__main__':
    app.run()

