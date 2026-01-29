from test_pkg.crowd_nav.policy_no_train.policy_factory import policy_factory
from test_pkg.crowd_nav.policy.cadrl import CADRL
from test_pkg.crowd_nav.policy.lstm_rl import LstmRL
from test_pkg.crowd_nav.policy.sarl import SARL

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
