from evaluation.test import test
from training.train import train

session_name = train()
test(session_name=session_name, is_visualize=False)
# test(session_name='test', is_visualize=True)
