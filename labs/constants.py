from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

MAX_ROW_NUMBER_THRESHOLD = 50000
MIN_ROW_NUMBER_THRESHOLD = 300
CORR_COEF = 0.85
TARGET_VALUE_NAME = 'target_var'
TEST_SIZE = 0.3
NUMBER_OF_NEIGHBORS = 5
NUMBER_OF_PARTS_FOR_CROSS_VALIDATION = 5
MODELS = {
    'knn': KNeighborsClassifier,
    'tree': DecisionTreeClassifier,
    'boosting': GradientBoostingClassifier,
    'log_reg': LogisticRegression
}
