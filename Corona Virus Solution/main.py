from flask import Flask
app = Flask(__name__)
import pickle

# Open a file where you stored the pickled data
file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route('/')
def hello_world():

    # Code for Inference

    inputFeatures = [102,1,22,-1,1]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    return 'Hello, World!' + str(infProb)

if __name__=="__main__":
    app.run(debug=True)