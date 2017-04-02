from evaluation.test import test
from training.train import train

session_name = train()
test(session_name)

exit()
