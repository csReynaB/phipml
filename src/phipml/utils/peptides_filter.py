# ======================
# Standard library
# ======================
import random

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features_to_keep_ = pd.Index([])

    def fit(self, X, y=None):
        """
        Fit the transformer. This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, X):
        """
        Transform the input data by selecting only the features that passed
        the filtering criteria during fit.
        """

        # Ensure features_to_keep_ has been set during fit
        # if not self.runTransformFirst:
        #   raise RuntimeError("You must call fit before transform.")
        if self.features_to_keep_.empty:
            print(
                "Warning: If 'fit' was called, no peptide features were kept. \n"
                "Returning zero peptide features."
            )
            # return X

        return X[self.features_to_keep_]


class EntropyFilter(FeatureFilter):
    def __init__(self, threshold=0.4):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def binary_entropy(feature):
        """
        Static method to compute entropy for a given column (feature).
        """

        if feature.isnull().all():  # Check if the column is empty
            return 0
        if feature.nunique() == 1:  # Only one class present
            # print(f"This feature contains same value for all samples: {feature.name}")
            return 0
        # Calculate the probability of each class (0 and 1)
        p_data = feature.value_counts(normalize=True)

        ##low entropy value, indicating low uncertainty because the outcome is mostly predictable (the majority class is known).
        ##neither class dominates significantly, leading to higher entropy, indicating greater uncertainty about the outcome
        return entropy(p_data, base=2)

    def fit(self, X, y=None):
        """
        Fit the transformer by calculating entropy for each feature
        and selecting features that meet the threshold.
        """
        entropies = X.apply(EntropyFilter.binary_entropy)  # (self.binary_entropy)
        self.features_to_keep_ = entropies[entropies >= self.threshold].index
        return self


class ClusteringFilter(FeatureFilter):
    """
    Perform hierarchical clustering on the DataFrame and select representative features randomly or using the medoid.

    Parameters:
    X (pd.DataFrame): Input DataFrame where rows are samples and columns are features.
    threshold (float): Threshold for forming flat clusters. Default is 0.2.
    dist (str): Distance metric to use. Default is 'hamming'.
    linkage_method (str): Linkage method to use ('average', 'single', 'complete', etc.). Default is 'average'.
    random_features (bool): Choose feature within cluster randomly or based on medoid
    seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
    features_to_keep_ (pd.index): Containing only the selected representative features.
    """

    def __init__(
        self,
        threshold=0.2,
        dist="hamming",
        linkage_method="average",
        random_features=False,
        seed=None,
    ):
        super().__init__()
        self.threshold = threshold
        self.dist = dist
        self.linkage_method = linkage_method
        self.random_features = random_features
        self.seed = seed

    # noinspection PyTypeChecker
    def fit(self, X, y=None):

        # Reset features_to_keep_ for the current fit
        list_features_to_keep_ = []

        # Compute pairwise distances and perform hierarchical clustering
        distance_matrix_local = pdist(X.T, metric=self.dist)
        z = linkage(distance_matrix_local, method=self.linkage_method)
        clusters_local = fcluster(z, self.threshold, criterion="distance")

        # Map clusters to features
        cluster_dict = {}
        for idx, cluster in enumerate(clusters_local):
            feature = X.columns[idx]
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(feature)

        if self.random_features:
            if self.seed is not None:
                random.seed(self.seed)
            self.features_to_keep_ = [
                random.choice(features)
                for features in cluster_dict.values()
                if features
            ]
        else:
            for features in cluster_dict.values():
                if len(features) == 1:
                    list_features_to_keep_.append(features[0])
                else:
                    sub_df = X[features].T
                    intra_cluster_distances = pdist(sub_df, metric=self.dist)
                    intra_cluster_distances_matrix = squareform(intra_cluster_distances)
                    sum_of_distances = intra_cluster_distances_matrix.sum(axis=0)
                    medoid_index = np.argmin(sum_of_distances)
                    list_features_to_keep_.append(features[medoid_index])

        self.features_to_keep_ = pd.Index(list_features_to_keep_)
        return self


class PrevalenceFilter(FeatureFilter):
    """
    Filters features (columns) in the DataFrame based on their prevalence across all samples.

    Parameters:
    X (pd.DataFrame): The input DataFrame where each column represents a feature (e.g., a protein)
                       and each row represents a sample.
    threshold (float): The minimum percentage of samples in which a feature must be present
                                  to be retained in the filtered DataFrame. Should be between 0 and 100.

    Returns:
    features_to_keep_ (pd.index): Containing only the features that meet the prevalence threshold.
    """

    def __init__(self, threshold_min=0.0, threshold_max=100.0):
        super().__init__()
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def fit(self, X, y=None):
        # Calculate the minimum and maximum number of samples a feature must be present in to meet the thresholds
        min_samples_required = (len(X.index) * self.threshold_min) / 100
        max_samples_allowed = (len(X.index) * self.threshold_max) / 100

        # Calculate the number of samples where each feature is non-zero and non-NA
        feature_counts = (X.notna() & (X > 0)).sum()

        # Filter based on both the minimum and maximum sample thresholds
        self.features_to_keep_ = feature_counts[
            (feature_counts >= min_samples_required)
            & (feature_counts <= max_samples_allowed)
        ].index

        return self


class Dictifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        # This transformer doesn't need to learn anything, so we just return self
        return self

    @staticmethod
    def transform(X):
        # Convert the DataFrame to a list of dictionaries
        return X.to_dict(orient="records")


class CorrelationFilter(FeatureFilter):
    def __init__(self, threshold=0.8, method="phi"):
        """
        Initialize the CorrelationFilter with a threshold for correlation and a method (pearson or phi).

        Parameters:
        - threshold: Correlation threshold to consider for dropping features.
        - method: Correlation method ('pearson' or 'phi').
        """
        super().__init__()
        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        """
        Fit the transformer by identifying features to drop based on the correlation matrix.

        Parameters:
        - X: DataFrame of features (samples as rows, features as columns).
        """
        # Calculate the correlation matrix (Pearson or phi)
        if self.method == "pearson":
            correlation_matrix = X.corr()
        elif self.method == "phi":
            correlation_matrix = self._phi_correlation(X)
        else:
            raise ValueError("Method should be 'pearson' or 'phi'")

        # Find highly correlated pairs
        highly_correlated_pairs = np.where(np.abs(correlation_matrix) > self.threshold)

        # Set of features to drop
        to_drop = set()
        for feature1, feature2 in zip(
            correlation_matrix.index[highly_correlated_pairs[0]],
            correlation_matrix.columns[highly_correlated_pairs[1]],
        ):
            if feature1 != feature2 and feature2 not in to_drop:
                to_drop.add(feature2)

        # Identify features to keep (i.e., those that are not in to_drop)
        self.features_to_keep_ = X.columns.difference(to_drop)

        return self

    @staticmethod
    def _phi_correlation(X):
        """
        Calculate the phi correlation matrix for binary data efficiently using vectorized operations.

        Parameters:
        - X: DataFrame of binary features.

        Returns:
        - DataFrame: Phi correlation matrix.
        """
        # Convert DataFrame to NumPy array
        X_np = X.values

        # Calculate the pairwise product of features
        n_11 = np.dot(X_np.T, X_np)
        n_10 = np.dot(X_np.T, 1 - X_np)
        n_01 = np.dot((1 - X_np).T, X_np)
        n_00 = np.dot((1 - X_np).T, 1 - X_np)

        # Calculate the numerator and denominator for the phi coefficient
        numerator = n_11 * n_00 - n_10 * n_01
        denominator = np.sqrt(
            (n_11 + n_10) * (n_01 + n_00) * (n_11 + n_01) * (n_10 + n_00)
        )

        # Compute the phi coefficient, handling division by zero
        # phi_matrix = np.where(denominator != 0, numerator / denominator, 0)
        phi_matrix = np.divide(numerator, denominator, where=denominator != 0)

        # Convert the matrix to a DataFrame
        return pd.DataFrame(phi_matrix, index=X.columns, columns=X.columns)
