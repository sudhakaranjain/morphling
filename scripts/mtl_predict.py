import pickle

with open("../../model.p3", "rb") as f:
  model = pickle.load(f)

print(model.predict_udpos('../../../data/udpos/train-af.tsv'))  # replace file path accordingly
