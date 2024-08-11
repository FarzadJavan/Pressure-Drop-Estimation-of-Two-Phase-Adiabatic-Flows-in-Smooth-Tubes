from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

exported_pipeline = make_pipeline(
    PCA(iterated_power=3, svd_solver="randomized"),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.9500000000000001, min_samples_leaf=15, min_samples_split=8, n_estimators=100)),
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    RidgeCV()
)

# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 12)
