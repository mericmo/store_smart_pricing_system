import optuna
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
warnings.filterwarnings('ignore')


class ModelOptimizer:
    """模型参数优化器"""

    def __init__(self, trainer_instance):
        """
        初始化模型优化器

        Args:
            trainer_instance: SalesPredictor实例
        """
        self.trainer = trainer_instance
        self.best_params = {}
        self.optimization_results = {}

    def prepare_cv_data(self, features_df, target_col='销售数量'):
        """
        准备交叉验证数据

        Returns:
            X, y, feature_columns
        """
        X_train, X_test, y_train, y_test, feature_columns = self.trainer._get_features_for_training(
            features_df, target_col
        )

        # 合并数据用于交叉验证
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])

        return X, y, feature_columns

    def time_series_cv_split(self, X, y, n_splits=5):
        """
        时间序列交叉验证分割

        Args:
            X: 特征数据
            y: 目标数据
            n_splits: 分割数

        Yields:
            训练集和验证集索引
        """
        n_samples = len(X)
        fold_size = n_samples // n_splits

        for i in range(n_splits - 1):
            train_end = (i + 1) * fold_size
            val_start = train_end
            val_end = val_start + fold_size

            train_indices = list(range(0, train_end))
            val_indices = list(range(val_start, min(val_end, n_samples)))

            yield train_indices, val_indices

    def optimize_lightgbm(self, features_df, target_col='销售数量',
                          n_trials=50, cv_folds=5, random_state=42):
        """
        优化LightGBM参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', -1, 50),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'num_boost_round': trial.suggest_int('num_boost_round', 100, 2000),
            }

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=params.pop('num_boost_round', 1000),
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 创建Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        # 添加固定参数
        best_params['objective'] = 'regression'
        best_params['metric'] = 'rmse'
        best_params['verbose'] = -1

        print(f"\nLightGBM参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['lightgbm'] = best_params
        self.optimization_results['lightgbm'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_xgboost(self, features_df, target_col='销售数量',
                         n_trials=50, cv_folds=5, random_state=42):
        """
        优化XGBoost参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': random_state,
                'n_jobs': -1
            }

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False,
                    early_stopping_rounds=50
                )

                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 创建Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\nXGBoost参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['xgboost'] = best_params
        self.optimization_results['xgboost'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_random_forest(self, features_df, target_col='销售数量',
                               n_trials=30, cv_folds=5, random_state=42):
        """
        优化随机森林参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': random_state,
                'n_jobs': -1
            }

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model = RandomForestRegressor(**params)
                model.fit(X_train_fold, y_train_fold)

                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\n随机森林参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['random_forest'] = best_params
        self.optimization_results['random_forest'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_gradient_boosting(self, features_df, target_col='销售数量',
                                   n_trials=30, cv_folds=5, random_state=42):
        """
        优化梯度提升参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'random_state': random_state
            }

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model = GradientBoostingRegressor(**params)
                model.fit(X_train_fold, y_train_fold)

                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\n梯度提升参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['gradient_boosting'] = best_params
        self.optimization_results['gradient_boosting'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_linear_regression(self, features_df, target_col='销售数量',
                                   n_trials=20, cv_folds=5, random_state=42):
        """
        优化线性回归参数（主要是正则化参数）

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            # 线性回归没有太多可调参数，主要是是否标准化
            normalize = trial.suggest_categorical('normalize', [True, False])

            # 创建模型
            model = LinearRegression(normalize=normalize)

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 创建Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\n线性回归参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['linear_regression'] = best_params
        self.optimization_results['linear_regression'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_ridge(self, features_df, target_col='销售数量',
                       n_trials=30, cv_folds=5, random_state=42):
        """
        优化岭回归参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            # 岭回归的主要参数
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'normalize': trial.suggest_categorical('normalize', [True, False]),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky',
                                                               'lsqr', 'sparse_cg', 'sag', 'saga']),
                'random_state': random_state
            }

            # 创建模型
            model = Ridge(**params)

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 创建Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\n岭回归参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['ridge'] = best_params
        self.optimization_results['ridge'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_lasso(self, features_df, target_col='销售数量',
                       n_trials=30, cv_folds=5, random_state=42):
        """
        优化Lasso回归参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            # Lasso回归的主要参数
            params = {
                'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'normalize': trial.suggest_categorical('normalize', [True, False]),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                'random_state': random_state
            }

            # 创建模型
            model = Lasso(**params, max_iter=10000)  # 增加最大迭代次数

            # 时间序列交叉验证
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 创建Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\nLasso回归参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['lasso'] = best_params
        self.optimization_results['lasso'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_svr(self, features_df, target_col='销售数量',
                     n_trials=30, cv_folds=3, random_state=42):
        """
        优化支持向量回归参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials: 试验次数（SVR较慢，减少试验次数）
            cv_folds: 交叉验证折数（SVR较慢，减少折数）
            random_state: 随机种子

        Returns:
            最佳参数和最佳分数
        """
        X, y, feature_columns = self.prepare_cv_data(features_df, target_col)

        def objective(trial):
            # SVR的主要参数
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])

            params = {
                'kernel': kernel,
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }

            # 如果选择poly核，添加degree参数
            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)

            # 创建模型
            model = SVR(**params)

            # 时间序列交叉验证（SVR较慢，使用较少折数）
            cv_scores = []

            for train_idx, val_idx in self.time_series_cv_split(X, y, cv_folds):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 创建Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        print(f"\n支持向量回归参数优化完成!")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"最佳参数: {best_params}")

        self.best_params['svr'] = best_params
        self.optimization_results['svr'] = {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }

        return best_params, best_score

    def optimize_all_models(self, features_df, target_col='销售数量',
                            n_trials_per_model=30, cv_folds=5, random_state=42):
        """
        优化所有支持模型的参数

        Args:
            features_df: 特征数据
            target_col: 目标列名
            n_trials_per_model: 每个模型的试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子

        Returns:
            所有模型的最佳参数
        """
        print("开始优化所有模型的参数...")

        results = {}

        # 优化LightGBM
        if 'lightgbm' in self.trainer.supported_algorithms:
            print("\n=== 优化LightGBM参数 ===")
            try:
                best_params, best_score = self.optimize_lightgbm(
                    features_df, target_col, n_trials_per_model, cv_folds, random_state
                )
                results['lightgbm'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"LightGBM优化失败: {e}")

        # 优化XGBoost
        if 'xgboost' in self.trainer.supported_algorithms:
            print("\n=== 优化XGBoost参数 ===")
            try:
                best_params, best_score = self.optimize_xgboost(
                    features_df, target_col, n_trials_per_model, cv_folds, random_state
                )
                results['xgboost'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"XGBoost优化失败: {e}")

        # 优化随机森林
        if 'random_forest' in self.trainer.supported_algorithms:
            print("\n=== 优化随机森林参数 ===")
            try:
                best_params, best_score = self.optimize_random_forest(
                    features_df, target_col, n_trials_per_model, cv_folds, random_state
                )
                results['random_forest'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"随机森林优化失败: {e}")

        # 优化梯度提升
        if 'gradient_boosting' in self.trainer.supported_algorithms:
            print("\n=== 优化梯度提升参数 ===")
            try:
                best_params, best_score = self.optimize_gradient_boosting(
                    features_df, target_col, n_trials_per_model, cv_folds, random_state
                )
                results['gradient_boosting'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"梯度提升优化失败: {e}")

        # 优化线性回归
        if 'linear_regression' in self.trainer.supported_algorithms:
            print("\n=== 优化线性回归参数 ===")
            try:
                # 线性回归参数较少，减少试验次数
                best_params, best_score = self.optimize_linear_regression(
                    features_df, target_col, min(10, n_trials_per_model), cv_folds, random_state
                )
                results['linear_regression'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"线性回归优化失败: {e}")

        # 优化岭回归
        if 'ridge' in self.trainer.supported_algorithms:
            print("\n=== 优化岭回归参数 ===")
            try:
                best_params, best_score = self.optimize_ridge(
                    features_df, target_col, n_trials_per_model, cv_folds, random_state
                )
                results['ridge'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"岭回归优化失败: {e}")

        # 优化Lasso回归
        if 'lasso' in self.trainer.supported_algorithms:
            print("\n=== 优化Lasso回归参数 ===")
            try:
                best_params, best_score = self.optimize_lasso(
                    features_df, target_col, n_trials_per_model, cv_folds, random_state
                )
                results['lasso'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"Lasso回归优化失败: {e}")

        # 优化支持向量回归（SVR较慢，减少试验次数）
        if 'svr' in self.trainer.supported_algorithms:
            print("\n=== 优化支持向量回归参数 ===")
            try:
                # SVR训练较慢，减少试验次数和交叉验证折数
                best_params, best_score = self.optimize_svr(
                    features_df, target_col,
                    min(15, n_trials_per_model),  # SVR较慢，减少试验次数
                    min(3, cv_folds),  # SVR较慢，减少折数
                    random_state
                )
                results['svr'] = {'params': best_params, 'score': best_score}
            except Exception as e:
                print(f"支持向量回归优化失败: {e}")

        print("\n所有模型参数优化完成!")
        self.print_optimization_summary()

        return results

    def print_optimization_summary(self):
        """打印优化结果摘要"""
        print("\n" + "=" * 50)
        print("参数优化结果摘要")
        print("=" * 50)

        for model_name, result in self.optimization_results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  最佳RMSE: {result['best_score']:.4f}")
            print(f"  参数数量: {len(result['best_params'])}个")

        # 找到最佳模型
        if self.optimization_results:
            best_model = min(self.optimization_results.items(),
                             key=lambda x: x[1]['best_score'])[0]
            best_score = self.optimization_results[best_model]['best_score']
            print(f"\n🏆 最佳模型: {best_model} (RMSE: {best_score:.4f})")

        print("=" * 50)