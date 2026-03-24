from pathlib import Path
import sys
import types
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, 
    log_loss, mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Tuple, Optional, List, Union
import optuna
import joblib


EXTS = ['png', 'pdf', 'svg']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

class BehavioralAnalysisGBM:
    """
    A class for analyzing behavioral data using XGBoost classification.
    """
    def __init__(
        self,
        save_path: Optional[Path] = None,
        random_state: int = 123,
        mode: str = 'classification'  # 'classification' or 'regression'
    ):
        self.save_path = save_path or Path.cwd() / 'results'
        self.random_state = random_state
        self.mode = mode
        self.model = None
        self.shap_values = None
        self.feature_names = None
        self.explainer = None
        self.metrics = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        outcome_col: str = 'Outcome',
        categorical_cols: Optional[List[str]] = None,
        test_size: float = 0.33
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Prepare data for analysis by handling categorical variables and splitting into train/test sets.
        """
        # Create copy to avoid modifying original
        df_use = df.copy()
        
        # Handle categorical columns
        cat_mappings = {}
        if categorical_cols:
            for col in categorical_cols:
                if col in df_use.columns:
                    # mapping = {val: idx for idx, val in enumerate(np.sort(df_use[col].unique()))}
                    unique_vals = df_use[col].unique()
            
                    try:
                        sorted_vals = np.sort(unique_vals)
                    except TypeError:
                        # Can't sort (mixed types), use original order
                        sorted_vals = unique_vals
                    
                    mapping = {val: idx for idx, val in enumerate(sorted_vals)}
                    df_use[col] = df_use[col].map(mapping)
                    cat_mappings[col] = mapping

        # Split features and target
        X = df_use.drop(columns=[outcome_col])
        y = df_use[outcome_col]
        self.feature_names = X.columns.tolist()

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if self.mode == 'classification' else None,
        )

        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test, cat_mappings

    def load_parameters(self, param_path: Optional[Path] = None, filename: str = 'best_params.npy') -> Dict:
        """
        Load previously saved hyperparameters.
        
        Args:
            param_path: Path to the saved parameters. If None, will look in default location.
            
        Returns:
            Dict of parameters
        """
        if param_path is None:
            param_path = self.save_path / 'optuna_studies' / filename
            
        if not param_path.exists():
            raise FileNotFoundError(f"No saved parameters found at {param_path}")
            
        try:
            params = np.load(param_path, allow_pickle=True).item()
            print(f"Loaded parameters from {param_path}")
            return params
        except Exception as e:
            raise Exception(f"Error loading parameters: {str(e)}")

    def save_parameters(self, params: Dict, filename: str = 'best_params.npy'):
        """
        Save parameters to file.
        """
        save_dir = self.save_path / '_optuna_studies'
        save_dir.mkdir(exist_ok=True, parents=True)
        param_path = save_dir / filename
        np.save(param_path, params)
        print(f"Saved parameters to {param_path}")

    def train_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Train the XGBoost model and return performance metrics.
        """
        X_train_smote, y_train_smote = X_train, y_train

        default_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if self.mode == 'classification':
            default_params['objective'] = 'binary:logistic'
            model_class = xgb.XGBClassifier
            default_params['eval_metric'] = 'logloss'
        else:
            default_params['objective'] = 'reg:squarederror'
            model_class = xgb.XGBRegressor
            default_params['eval_metric'] = 'rmse'
        
        if params:
            default_params.update(params)

        # Add early stopping parameters
        default_params['early_stopping_rounds'] = 100

        self.model = model_class(**default_params)
        
        # Create evaluation set
        eval_set = [(X_test, y_test)]
        self.model.fit(
            X_train_smote,
            y_train_smote,
            eval_set=eval_set,
            verbose=False
        )

        # Calculate metrics based on mode
        y_pred = self.model.predict(X_test)
        if self.mode == 'classification':
            metrics = {
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'balanced_accuracy_train': balanced_accuracy_score(y_train, self.model.predict(X_train)),
                'f1_score': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

        return metrics

    def calculate_feature_importance(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate permutation importance.
        """
       
        perm_result = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=100,
            random_state=self.random_state,
            scoring='neg_mean_squared_error' if self.mode == 'regression' else None
        )
        
        return perm_result

    def optimize_parameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 1000,
        save_study: bool = True,
        file_prefix: str = '',
    ) -> Dict:
        """
        Optimize model hyperparameters using Optuna.
        """
        def objective(trial, X_train, X_valid, y_train, y_valid):
            param_grid = {
                "subsample": trial.suggest_float("subsample", 0.1, 1),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 7, 20),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
                "gamma": trial.suggest_float("gamma", 0, 20),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            }

            if self.mode == 'classification':
                param_grid["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1, 10)
                param_grid["eval_metric"] = "logloss"
            else:
                param_grid["eval_metric"] = "rmse"

            # Add early stopping parameter
            param_grid["early_stopping_rounds"] = 100

            model = xgb.XGBClassifier(**param_grid) if self.mode == 'classification' else xgb.XGBRegressor(**param_grid)
            model.set_params(random_state=self.random_state, verbosity=0, n_jobs=20)

            # Create evaluation set
            eval_set = [(X_valid, y_valid)]
            
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False
            )

            preds = model.predict_proba(X_valid)[:, 1] if self.mode == 'classification' else model.predict(X_valid)
            return log_loss(y_valid, preds) if self.mode == 'classification' else mean_squared_error(y_valid, preds)

        # Create study
        sampler = optuna.samplers.TPESampler(seed=self.random_state) # For reproducibility
        study = optuna.create_study(
            direction="minimize", 
            study_name="XGBoost " + self.mode.capitalize(),
            sampler=sampler,
            )

        # Split data for optimization
        X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        X_train_smote, y_train_smote = X_train_opt, y_train_opt

        # Create optimization function
        func = lambda trial: objective(trial, X_train_smote, X_valid, y_train_smote, y_valid)

        # Run optimization
        study.optimize(func, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

        # Print results
        print("Number of finished trials: ", len(study.trials))
        print(f"Best value of - {'logloss' if self.mode == 'classification' else 'mse'}: {study.best_value:.5f}")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"\t{key}: {value}")

        # Save study if requested
        if save_study and self.save_path:
            study_path = self.save_path / 'optuna_studies'
            study_path.mkdir(exist_ok=True, parents=True)
            joblib.dump(study, study_path / f'{file_prefix}_{self.mode}_optuna_study.pkl')

        return study.best_params

    def run_analysis(
        self,
        df: pd.DataFrame,
        outcome_col: str = 'Outcome',
        categorical_cols: Optional[List[str]] = None,
        params: Optional[Dict] = None,
        param_path: Optional[Path] = None,
        optimize: bool = False,
        n_trials: int = 1000,
        interaction_features: Optional[List[Tuple[str, str]]] = None,
        save_prefix: str = 'analysis',
        file_prefix: str = '',
    ) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            df: Input DataFrame
            outcome_col: Name of the outcome column
            categorical_cols: List of categorical column names
            params: Dictionary of model parameters (optional)
            param_path: Path to saved parameters (optional)
            optimize: Whether to run parameter optimization
            n_trials: Number of optimization trials
            interaction_features: List of feature pairs to analyze interactions
            save_prefix: Prefix for saved files
        """
        # Prepare data
        X_train, X_test, y_train, y_test, cat_mappings = self.prepare_data(
            df,
            outcome_col,
            categorical_cols
        )
        
        # Handle parameters
        param_file = f'{file_prefix}_{self.mode}_best_params.npy'
        if optimize:
            print("Optimizing hyperparameters...")
            params = self.optimize_parameters(
                X_train,
                y_train,
                n_trials=n_trials,
                file_prefix=file_prefix,
            )
            # Save the optimized parameters
            self.save_parameters(params, filename=param_file)
            print("Optimization completed.")
        elif params is None:
            # Try to load parameters if not provided and not optimizing
            try:
                params = self.load_parameters(param_path, filename=param_file)
            except FileNotFoundError:
                print("No saved hyperparameters found. Using default parameters.")
                params = {}

        # Train model
        self.metrics = self.train_model(X_train, X_test, y_train, y_test, params)

        # Calculate feature importance
        X = pd.concat([X_train, X_test])

        # Use a subset of training data as background
        background_data = X_train.sample(min(100, len(X_train)), random_state=self.random_state)
        print(f"Starting SHAP explainer")
        
        # Ensure all data is float64
        background_data = background_data.astype(float)
        X = X.astype(float)
        
        # Create explainer with model output set to 'raw' for both classification and regression
        self.explainer = shap.TreeExplainer(
            self.model,
            data=background_data,
            feature_perturbation='interventional',
            model_output='raw'  # Always use 'raw' for both classification and regression
        )
        
        # Get SHAP values for all data points
        if self.mode == 'classification':
            # For classification, get class probabilities
            shap_values = self.explainer.shap_values(X)#[1] # Get values for positive class
            # Ensure shap_values is 2D with correct number of features
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, X.shape[1])
        else:
            # For regression, get raw predictions
            shap_values = self.explainer.shap_values(X)
            # Ensure shap_values is 2D with correct number of features
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, X.shape[1])
        print(f"SHAP explainer completed")
        self.shap_values = shap_values

        # Calculate feature importance
        perm_result = self.calculate_feature_importance(X_test, y_test)

        # Calculate feature importance scores
        feature_importance = pd.Series(
            np.abs(self.shap_values).mean(0),
            index=self.feature_names
        ).sort_values(ascending=False)

        # Store results for later use
        self.analysis_results = {
            'metrics': self.metrics,
            'feature_importance': feature_importance,
            'cat_mappings': cat_mappings,
            'model': self.model,
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'perm_result': perm_result,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'interaction_features': interaction_features,
            'file_prefix': file_prefix,
            'save_prefix': save_prefix,
        }

        return self.analysis_results

    def save_results(self, save_dir: Path, file_prefix: str = ''):
        """
        Save analysis results, including the analyzer object and figures.
        
        Args:
            save_dir: Directory to save results
            file_prefix: Prefix for saved files
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analyzer object
        analyzer_path = save_dir / f'{file_prefix}_analyzer.pkl'
        joblib.dump(self, analyzer_path)
        
        # Save analysis results
        results_path = save_dir / f'{file_prefix}_results.pkl'
        joblib.dump(self.analysis_results, results_path)
        
        # # Save figures
        # figures_dir = save_dir / 'figures'
        # figures_dir.mkdir(exist_ok=True)
        
        # # Save feature importance plot
        # if hasattr(self, 'analysis_results') and 'feature_importance' in self.analysis_results:
        #     plt.figure(figsize=(10, 6))
        #     self._plot_feature_importance(
        #         self.analysis_results['X_train'],
        #         self.analysis_results['X_test'],
        #         self.analysis_results['shap_values'],
        #         self.analysis_results['perm_result'],
        #         figures_dir,
        #         file_prefix
        #     )
        #     plt.close()
        
        print(f"Saved analysis results to {save_dir}")

    @classmethod
    def load_results(cls, load_dir: Path, file_prefix: str = '') -> Tuple['BehavioralAnalysisGBM', Dict]:
        """
        Load previously saved analysis results.
        
        Args:
            load_dir: Directory containing saved results
            file_prefix: Prefix of saved files
            
        Returns:
            Tuple of (analyzer object, analysis results)
        """
        load_dir = Path(load_dir)

        # Allow loading pickles saved when the class lived in explain_behaviour
        _mod = types.ModuleType('explain_behaviour.models.behavioural_analysis_GBM')
        _mod.BehavioralAnalysisGBM = cls
        sys.modules.setdefault('explain_behaviour', types.ModuleType('explain_behaviour'))
        sys.modules.setdefault('explain_behaviour.models', types.ModuleType('explain_behaviour.models'))
        sys.modules['explain_behaviour.models.behavioural_analysis_GBM'] = _mod

        # Load analyzer object
        analyzer_path = load_dir / f'{file_prefix}_analyzer.pkl'
        if not analyzer_path.exists():
            raise FileNotFoundError(f"No saved analyzer found at {analyzer_path}")
        analyzer = joblib.load(analyzer_path)
        
        # Load analysis results
        results_path = load_dir / f'{file_prefix}_results.pkl'
        if not results_path.exists():
            raise FileNotFoundError(f"No saved results found at {results_path}")
        results = joblib.load(results_path)
        
        # Restore results to analyzer
        analyzer.analysis_results = results
        
        print(f"Loaded analysis results from {load_dir}")
        return analyzer, results



def plot_figure1k(load_dir: Path, saving_path: Path, name: str = 'Figure1K', saving_formats: list = ['png', 'svg']):
    load_dir = Path(load_dir)
    analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)

    X_test = analyzer.X_test
    y_test = analyzer.y_test

    perm_results = permutation_importance(
        analyzer.model, X_test, y_test,
        n_repeats=10000,
        random_state=analyzer.random_state,
        scoring='balanced_accuracy',
        n_jobs=-1,
    )

    importances = perm_results.importances
    ci_lower = np.percentile(importances, 2.5, axis=1)
    ci_upper = np.percentile(importances, 97.5, axis=1)
    significant = (ci_lower > 0) | (ci_upper < 0)
    feature_names = X_test.columns if hasattr(X_test, 'columns') else np.arange(importances.shape[0])

    print("Significant features (95% CI does not include 0):")
    for name, lower, upper, sig in zip(feature_names, ci_lower, ci_upper, significant):
        if sig:
            print(f"{name}: 95% CI = [{lower:.4f}, {upper:.4f}]")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sorted_idx = perm_results.importances_mean.argsort()
    sig_idx = sorted_idx[significant[sorted_idx]]
    non_sig_idx = sorted_idx[~significant[sorted_idx]]

    ax.barh([feature_names[i] for i in non_sig_idx], perm_results.importances_mean[non_sig_idx],
            xerr=[perm_results.importances_mean[non_sig_idx] - ci_lower[non_sig_idx],
                  ci_upper[non_sig_idx] - perm_results.importances_mean[non_sig_idx]],
            color='lightgray', edgecolor='k', label='Not significant', alpha=0.7)
    ax.barh([feature_names[i] for i in sig_idx], perm_results.importances_mean[sig_idx],
            xerr=[perm_results.importances_mean[sig_idx] - ci_lower[sig_idx],
                  ci_upper[sig_idx] - perm_results.importances_mean[sig_idx]],
            color='tab:blue', edgecolor='k', label='Significant', alpha=0.9)

    ax.set_xlabel('Permutation Importance')
    ax.set_title('Permutation Importances', fontsize=10)
    ax.legend()
    fig.tight_layout()
    sns.despine(fig=fig)

    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)
    save_fig_name = saving_path / name
    for ext in saving_formats:
        fig.savefig(save_fig_name.with_suffix(f'.{ext}'), dpi=300)


def plot_figure1l(load_dir: Path, saving_path: Path, name: str = 'Figure1L', saving_formats: list = ['png', 'svg']):
    load_dir = Path(load_dir)
    analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)

    interaction_features = ['Whisker trial in block', 'Context']
    type_a = 'discrete'

    X_disp = pd.concat([analyzer.analysis_results['X_train'], analyzer.analysis_results['X_test']])
    shap_values = analyzer.analysis_results['shap_values']

    mapping = {8: -1, 7: -2, 6: -3, 5: -4}
    X_disp['Whisker trial in block'] = X_disp['Whisker trial in block'].astype(int)
    X_disp['Whisker trial in block'] = X_disp['Whisker trial in block'].map(mapping).fillna(X_disp['Whisker trial in block'])
    X_disp['Context'] = X_disp['Context'].astype(bool)

    def get_context_to_plot(row):
        if row['Whisker trial in block'] > 0:
            return row['Context']
        else:
            return not row['Context']

    X_disp['Context'] = X_disp.apply(get_context_to_plot, axis=1)

    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(-1, 1)

    feature_a, feature_b = interaction_features
    feature_a_index = X_disp.columns.get_loc(feature_a)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    custom_palette = {0: 'darkmagenta', 1: 'green'}

    if type_a == 'discrete':
        sns.violinplot(
            x=X_disp[feature_a],
            y=shap_values[:, feature_a_index],
            hue=X_disp[feature_b],
            palette=custom_palette,
            ax=ax,
            fill=False,
            inner="quart",
            linewidth=1,
            cut=0,
            density_norm='width',
        )
    elif type_a == 'continuous':
        sns.scatterplot(
            x=X_disp[feature_a],
            y=shap_values[:, feature_a_index],
            hue=X_disp[feature_b],
            palette=custom_palette,
            ax=ax,
            legend=False,
        )
    else:
        raise NotImplementedError

    handles = [plt.scatter([], [], c=color, label=label)
               for label, color in zip(['to W-', 'to W+'], custom_palette.values())]
    ax.legend(handles=handles)
    ax.set_xlabel(feature_a)
    ax.set_ylabel(f'SHAP value of \n{feature_a.lower()}')
    ax.axvline(x=3.5, c='k', linestyle='--')
    sns.despine(fig, offset=10)
    fig.tight_layout()

    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)
    save_fig_name = saving_path / name
    for ext in saving_formats:
        fig.savefig(save_fig_name.with_suffix(f'.{ext}'), dpi=300)
