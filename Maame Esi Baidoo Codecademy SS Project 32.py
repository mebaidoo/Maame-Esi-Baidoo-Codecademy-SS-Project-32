import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

import codecademylib3
np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())
#Selecting the pts column for where fran_id is knicks and for where fran_id is nets for 2010 to compare the two teams with respect to their points
knicks_pts = nba_2010.pts[nba.fran_id == "Knicks"]
nets_pts = nba_2010.pts[nba.fran_id == "Nets"]
#Calculating the difference between the two teams' average points
diff_means_2010 = np.mean(knicks_pts) - np.mean(nets_pts)
print("Average points difference for 2010: " + str(diff_means_2010))
#Creating a set of overlapping histograms that can be used to compare the points scored for the Knicks compared to the Nets and understand the mean difference more
#We are dealing with quantitative variable, pts and categorical variable, fran_id
plt.hist(knicks_pts, color = "blue", label = "Knicks", normed = True, alpha = 0.5)
plt.hist(nets_pts, color = "orange", label = "Nets", normed = True, alpha = 0.5)
plt.legend()
plt.show()
#Looks like there is some association between the team and points scored in 2010 from the overlapping histograms as they don't overlap much
plt.clf()
#Repeating the steps for 2014 to compare the two teams with respect to their points in 2014
knicks_pts = nba_2014.pts[nba.fran_id == "Knicks"]
nets_pts = nba_2014.pts[nba.fran_id == "Nets"]
diff_means_2014 = np.mean(knicks_pts) - np.mean(nets_pts)
print("Average points difference for 2014: " + str(diff_means_2014))
plt.hist(knicks_pts, color = "blue", label = "Knicks", normed = True, alpha = 0.5)
plt.hist(nets_pts, color = "orange", label = "Nets", normed = True, alpha = 0.5)
plt.legend()
plt.show()
#Looks like there is no association between the team and points scored in 2014 from the overlapping histograms as they overlap
plt.clf()
#Drawing side-by-side box plots for the 2010 data to compare the teams to the scores and see if there is an association there
sns.boxplot(data = nba_2010, x = "fran_id", y = "pts")
plt.show()
#It looks like fran_id and pts are associated in some way as some teams do not overlap in the box plot

#Analyzing the relationships between categorical variables in the 2010 data
#Determining if teams to win more games at home or away using a contingency table for the game_result and game_location columns
location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)
print(location_result_freq)
location_result_proportions = location_result_freq / len(nba_2010)
print(location_result_proportions)
#Getting the expected contingency table and chi-square statistic
chi_sq_value, pval, dof, expected_table = chi2_contingency(location_result_freq)
print("The expected contingency table:" + "\n" + str(expected_table))
print("The chi-square value: " + str(chi_sq_value))
#Looks like there is an association between the game location and game result but not very strong

#Analyzing relationships between quantitative variables in the data
#Calculating the covariance between forecast and point_diff in the 2010 data to determine if teams with a higher probability of winning (forecast)also tend to win games by more points (point_diff)
cov_forecast_point = np.cov(nba_2010.forecast, nba_2010.point_diff)
print("The covariance matrix:" + "\n" + str(cov_forecast_point))
#The covariance (1.37) is not very big, there may be an association but not strong.
#Calculating the correlation between forecast and point_diff to explore the association further
corr_forecast_point, p = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print("The correlation between forecast and the points difference is " + str(corr_forecast_point))
#There is a 44% correlation between forecast and points difference
#Generating a scatter plot of forecast and point_diff to explore the association further
plt.clf()
plt.scatter(x = nba_2010.forecast, y = nba_2010.point_diff)
plt.xlabel("Forecast Probability")
plt.ylabel("Points Difference")
plt.title("NBA Predictions")
plt.show()