import pandas as pd
import numpy as np

from hics.incremental_correlation import IncrementalCorrelation
from hics.result_storage import DefaultResultStorage
from hics.scored_slices import ScoredSlices
from hics.divergences import KS, KLD


target = 'zzz'
i='Levine_13dim'
names = [_ for _ in range(np.shape(data)[1])]
data = pd.DataFrame(data,  columns=names)
#data = data[:,markers_dict[i]]
input_features = data.columns.values[0:12]
storage = DefaultResultStorage(input_features)
correlation = IncrementalCorrelation(data, target = 12, result_storage=storage, iterations = 10, alpha = 0.1, categorical_features=None,
                 continuous_divergence=KS)


correlation.update_bivariate_relevancies(runs = 1)


correct_result = pd.DataFrame({'value' : [0, 1, 2], 'count' : [3, 1, 1], 'probability' : [0.6, 0.2, 0.2]})
dataset = pd.DataFrame({'target' : [1, 1, 1, 0, 0, 0, 2, 2, 2], 'feature' : [0, 1, 2, 3, 4, 5, 6, 7, 8]})
condition = {'feature' : 'feature', 'indices' : [2, 3, 4, 5, 6], 'from_value' : 2, 'to_value' : 6}
target = 'target'
test_HiCS = HiCS(dataset, alpha = 0.1, iterations = 100)
cond_dist = test_HiCS.calculate_conditional_distribution([condition], target)
self.assertTrue(cond_dist.equals(correct_result))

