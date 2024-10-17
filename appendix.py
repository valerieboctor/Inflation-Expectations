#==================================================================================================
#								inflation beta distribution fitting
#==================================================================================================

# In this script:
# 1.) define functions for distr. fitting via non-linear least squares

# Beta Distribution Fitting for 3+ Bins/ 3 Scenarios Responses
# 2.) fit 3+ bins PDF data to beta distribution, split by endpoint cases. Store values.
# 3.) fit 3 scenarios PDF data to beta distribution. Store values.
# 4.) Construct individual beta distr.-implied CDFs

# (Isosceles) Triangular Distribution Fitting for 2 Bins/ 2 Scenarios Reponses 
# 5.) fit 2-bins PDF data to triang distr., split by endpoint cases. Store values.
# 6.) fit 2-scenarios PDF data to triang distr., split by endpoint cases. Store values.
# 7. Construct individual triangular distr.-implied CDFs

# Uniform Distribution Fitting for 1 Bin Reponses 
# 8.) fit 1-bin responses to uniform distribution. Store values.
# 9.) Construct individual uniform distr.-impled CDFs.

#--------------------------------------------------------------------------------------------------
# import python modules
from __future__ import division
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import os
import scipy 
from scipy.stats import uniform
from scipy.stats import triang
from scipy.stats import beta
from scipy.optimize import least_squares
import time


os.chdir(os.path.expanduser("~/Dropbox/Expectations Survey Design/data"))
print("Current directory is "+os.getcwd())

fig_path = os.path.expanduser("~/Dropbox/Expectations Survey Design/tables_figures")
#--------------------------------------------------------------------------------------------------
# 			1.) define functions for parametric distribution fitting
#--------------------------------------------------------------------------------------------------

# 								1.1 define uniform distr. function
#---------------------------------------------------------------------------------------------------
def y_unif_3(theta,t): 
	#case 3: θ = ({}), l & r known (s is known)
	pdf = uniform.pdf(t, loc= l, scale=(r-l)) 
	return pdf

# 							1.2 define triangular distr. functions
#---------------------------------------------------------------------------------------------------
# triangular distribution has 3 parameters: θ=(c,l,s)
# θ_0 = c: shape (assumed to be .5, which implies each distr. is isosceles (following SCE))
# θ_1 = l: location, i.e, left endpoint
# θ_2 = s: scale (s = r - l)

# there are three relevant cases
# 1. θ = (l); r known, estimating l is sufficient to pin down s.
# 2. θ = (s); l known
# 3. θ = ({}); l,r known.

# Define PDF functions for each case
def y_triang_1(theta,t): 
 	#case 1: θ = (l), r known 
 	pdf = triang.pdf(t, .5, loc = theta[0], scale=(r - theta)) 
 	return pdf

def y_triang_2(theta,t): 
	#case 2: θ = (s), l known
	pdf = triang.pdf(t, .5, loc = l, scale = theta)
	return pdf

def y_triang_3(theta,t): 
	#case 3: θ = ({}), l & r known (s is known)
	pdf = triang.pdf(t, .5, loc = l, scale=(r-l)) 
	return pdf


# 							1.3 define generalized beta distr. functions
#---------------------------------------------------------------------------------------------------

# generalized beta distribution has 4 parameters: θ=(α,β,l,s)
# θ_0 = α: shape 
# θ_1 = β: shape
# θ_2 = l: location (shift X by l), minimum value reported by individual
# θ_3 = s: scale , difference between max, min value reported by individual

# For responses where the outer bins aren't used, l and r are well-defined. For cases where respondents do use the outer bins, we want to estimate l and/or r simultaneously with α & β. 

# The pdf f(t;θ)~beta is a function of parameters, θ, and the inflation values, t.
# define f(t;θ) for 4 cases: 
# 0. θ = (α,β,l,s); neither endpoint known
# 1. θ = (α,β,l); right endpoint known
# 2. θ = (α,β,s); left endpoint known 
# 3. θ = (α,β); both endpoints known

# Define PDF functions for each case
def y_beta_0(theta,t): #only defined for estimating both endpoints
 	#case 0: θ = (α,β,l,r) 
 	pdf = beta.pdf(t, theta[0], theta[1], loc= theta[2], scale= (theta[3]))
 	return pdf

def y_beta_1(theta,t): 
 	#case 1: θ = (α,β,l), r known 
 	pdf = beta.pdf(t, theta[0], theta[1], loc= theta[2], scale=(r - theta[2])) 
 	return pdf

def y_beta_2(theta,t): 
	#case 2: θ = (α,β,s), l known
	pdf = beta.pdf(t, theta[0], theta[1], loc= l, scale=(theta[2]))
	return pdf

def y_beta_3(theta,t): 
	#case 3: θ = (α,β), l & r known (s is known)
	pdf = beta.pdf(t, theta[0], theta[1], loc= l, scale=(r-l)) 
	return pdf

# 							1.4 Define additional functions
#---------------------------------------------------------------------------------------------------
# fun_beta returns an array of residuals from the parametric fit pdf minus the sample data points	# this is required for the least_squares routine in SciPy
def fun(theta):
    return y_distr(theta, ts) - ys # y_distr is specified as any of the y's defined above.

# Define uniform distr summary stat (unif-implied mean, median, uncertainty) function
def unif_sum_stats(theta_hat): #format theta_hat as a 2-tuple containing l, s.
	mean = uniform.mean(theta_hat[0], theta_hat[1])
	median = uniform.median(theta_hat[0], theta_hat[1])
	std = uniform.std(theta_hat[0], theta_hat[1])
	sum_stats = [mean, median, std]
	return sum_stats

# Define triangular distr summary stat (triang -implied mean, median, uncertainty) function
def triang_sum_stats(theta_hat): #format theta_hat as a 3-tuple containing c, l, s.
	mean = triang.mean(theta_hat[0], theta_hat[1], theta_hat[2])
	median = triang.median(theta_hat[0], theta_hat[1], theta_hat[2])
	std = triang.std(theta_hat[0], theta_hat[1], theta_hat[2])
	sum_stats = [mean, median, std]
	return sum_stats

# Define beta summary stat (beta-implied mean, median, uncertainty) function
def beta_sum_stats(theta_hat): #format theta_hat as a 4 tuple containing a, b, l,s.
	mean = beta.mean(theta_hat[0], theta_hat[1], theta_hat[2], theta_hat[3])
	median = beta.median(theta_hat[0], theta_hat[1], theta_hat[2], theta_hat[3])
	std = beta.std(theta_hat[0], theta_hat[1], theta_hat[2], theta_hat[3])
	sum_stats = [mean, median, std]
	return sum_stats

#---------------------------------------------------------------------------------------------------
# 			2.) fit 3+ bins data to beta distribution, store values
#---------------------------------------------------------------------------------------------------
# read in bins-based empirical pdf data
df = pd.read_stata('v1_pdfs_cases.dta', columns = ["id","type", "case", "bin_midpoint", "epi_12m_", "v1_bin_min_i", "v1_bin_max_i"]) 
df = df[df.type == 3] # type 3 since fitting to beta distr.
# keep obs with positive probability
df = df[df.epi_12m_ > 0] 
# individual data is divided into four cases (See stata code): 
# 0. neither endpoint known 
# 1. only right endpoint known
# 2. only left endpoint known 
# 3. both endpoints known

# case 0: θ = (α,β,l,s)
df_0 = df[df.case == 0]
# create a new id variable with no gaps after dropping outliers.
df_0.insert(1, "id2", df_0.groupby(['id']).ngroup())

# case 1: θ = (α,β,l)
df_1 = df[df.case ==1] 
df_1.insert(1, "id2", df_1.groupby(['id']).ngroup())

# case 2: θ = (α,β,s)
df_2 = df[df.case ==2] 
df_2.insert(1, "id2", df_2.groupby(['id']).ngroup())

# case 3: θ = (α,β)
df_3 = df[df.case == 3]
df_3.insert(1, "id2", df_3.groupby(['id']).ngroup())

#  create empty dataframe to store theta estimates
df_beta_bins = pd.DataFrame()
df_beta_bins['id'] = []
df_beta_bins['case'] = []
df_beta_bins['alpha'] = []
df_beta_bins['beta'] = []
df_beta_bins['location'] = []
df_beta_bins['scale'] = []
df_beta_bins['mean'] = []
df_beta_bins['median'] = []
df_beta_bins['std'] = []
df_beta_bins.head()

# fit individual pdf data to beta distribution using non-linear least squares 

cases = [df_0, df_1, df_2, df_3]
# initial guess for theta by case:
theta0_cases= [[2,3,-16,32],[2,3,-16],[2,3,32],[2,3]]

location_lb = -100
scale_ub =  np.inf

# bounds on θ vary by case:
bounds_cases = [	[[-np.inf, -np.inf, location_lb, -np.inf], [np.inf, np.inf, np.inf, scale_ub]], [[-np.inf, -np.inf, location_lb],[np.inf, np.inf, np.inf]], [[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, scale_ub]], [[-np.inf, -np.inf], [np.inf, np.inf]]  	]

# fit individual pdfs to beta distribution, looping over cases, then individuals within case
# make switch case function that returns y_beta_`case' for each case.
for c in range(len(cases)):
	#store number of unique ids to iterate over
	cases[c] = cases[c].sort_values(by = ["id2", "bin_midpoint"])
	case = np.amax(cases[c].case).astype('int')
	obs = np.amax(cases[c].id2) + 1
	for i in range(obs):
		temp_df = cases[c][cases[c].id2 ==i] 
		i_1 = i+1
		r = np.amax(temp_df.v1_bin_max_i) # only relevant for cases 1,3
		l = np.amax(temp_df.v1_bin_min_i) # only relevant for cases 2,3
		print("Bins Case "+str(c)+": Beta fitting for respondent "+str(i_1)+" out of "+str(obs))
		ys = np.asarray(temp_df.epi_12m_) #probabilities 
		ts = np.asarray(temp_df.bin_midpoint) #point values
		theta0 = theta0_cases[c]
		if case == 0:
			y_distr = y_beta_0
		if case == 1:
			y_distr = y_beta_1
		if case == 2:
			y_distr = y_beta_2
		if case == 3:
			y_distr = y_beta_3
		# fit data to beta distribution using non-linear least squares
		res = least_squares(fun, theta0, method = 'trf', bounds = bounds_cases[c])
		# calculate implied mean, median, and uncertainty of inflation expectations at the individual level
		if case == 0: #estimate location and scale 
			theta_hat_i = res.x
		if case == 1: #estimate location, not scale
			theta_hat_i = [res.x[0], res.x[1], res.x[2], r - res.x[2]]
		if case == 2: #estimate scale, not location
			theta_hat_i = [res.x[0], res.x[1], l, res.x[2]]
		if case == 3: #estimate shape parameters only.
			theta_hat_i = [res.x[0], res.x[1], l, r - l]
		# generate individual-level summary stats, parameter estimates. Store values.
		sum_stats_i = beta_sum_stats(theta_hat_i)
		df_beta_bins = df_beta_bins.append({'id' : np.amax(temp_df.id), 'case' : np.amax(temp_df.case), 'alpha' : theta_hat_i[0], 'beta': theta_hat_i[1],'location' : theta_hat_i[2], 'scale': theta_hat_i[3], 'mean': sum_stats_i[0], 'median':sum_stats_i[1], 'std':sum_stats_i[2]}, ignore_index = True)
df_beta_bins.insert(1, "id2", df_beta_bins.groupby(['id']).ngroup())
df_beta_bins.to_csv('v1_beta_fit.csv')
############# example #############
# ys = np.array([ .1, .1, .2, .15, .35, .05, .05]) #probabilities 
# ts = np.array([-14, -10, 1, 3, 6, 10, 14]) #corresponding bins (midpoints)
# theta0 = [1,1,-16, 54] #initial guess for theta 
# y_distr = y_beta_0 #use the 0th case form of the pdf, in which theta is a 4-tuple.
# res = least_squares(fun, theta0, bounds = (-np.inf, np.inf), method = 'trf' )
# #res.x stores the parameter estimates
# #res.fun stores residuals from fitted data minus ys. 
# y_fitted = res.fun + ys
# plt.plot(ts, y_fitted) 
# plt.plot(ts, ys)

#--------------------------------------------------------------------------------------------------
# 					3.) fit 3-scenarios data to beta distribution, store values
#--------------------------------------------------------------------------------------------------
df_scen = pd.read_stata('v2_pdfs_cases.dta')
df_scen = df_scen[df_scen.type == 3]
df_scen.insert(1, "id2", df_scen.groupby(['id']).ngroup())

# create empty dataframe to store parameter estimates
df_beta_scen = pd.DataFrame()
df_beta_scen['id'] = []
df_beta_scen['case'] = []
df_beta_scen['alpha'] = []
df_beta_scen['beta'] = []
df_beta_scen['location'] = []
df_beta_scen['scale'] = []
df_beta_scen['mean'] = []
df_beta_scen['median'] = []
df_beta_scen['std'] = []

theta0 = [2,4,0,40]
obs = np.amax(df_scen.id2) + 1
location_lb = -100
scale_ub = np.inf
bounds_scen = [-np.inf, -np.inf, location_lb, -np.inf],[np.inf, np.inf, np.inf, scale_ub]
for i in range(obs):
	temp_df = df_scen[df_scen.id2 ==i] 
	i_1 = i+1 
	print("Version 2: Beta fitting for respondent "+str(i_1)+" out of "+str(obs))
	ys = np.asarray(temp_df.prob) #probabilities 
	ts = np.asarray(temp_df.point) #point values
	y_distr = y_beta_0
	res = least_squares(fun, theta0, method = 'trf', bounds = bounds_scen)
	theta_hat_i = res.x
	# generate individual-level sum stats. Store values
	sum_stats_i = beta_sum_stats(theta_hat_i)
	df_beta_scen = df_beta_scen.append({'id' : np.amax(temp_df.id), 'case' : 0, 'alpha' : res.x[0], 'beta':res.x[1],'location' : res.x[2], 'scale':res.x[3], 'mean': sum_stats_i[0], 'median':sum_stats_i[1], 'std':sum_stats_i[2]}, ignore_index = True)
df_beta_scen.insert(1, "id2", df_beta_scen.groupby(['id']).ngroup())
df_beta_scen.to_csv('v2_beta_fit.csv')

#--------------------------------------------------------------------------------------------------
# 					4.) Construct and store individual beta distr-implied CDFs
#--------------------------------------------------------------------------------------------------
grid_min = -20
grid_max = 40
n_points = (grid_max - grid_min +1)
epi  = np.linspace(grid_min, grid_max, n_points)

# Construct bins and scenarios beta-implied CDFs
# create empty dataframe to store CDF values 
df_cdfs = pd.DataFrame()
df_cdfs['id'] = []
df_cdfs['point'] = []
df_cdfs['prob_mass'] = []
df_cdfs['version'] = []

version = [1,2]
beta_data = [df_beta_bins, df_beta_scen]
obs = [np.amax(df_beta_bins.id2) + 1, np.amax(df_beta_scen.id2) + 1]

for v in range(len(version)):
	obs_v = obs[v]
	for i in range(obs_v):
		temp_df = beta_data[v][beta_data[v].id2 == i]
		i_1 = i+1
		theta_hat = np.asarray(temp_df[['alpha', 'beta', 'location', 'scale']])[0]
		temp_df.head()
		cdf_i = beta.cdf(epi, theta_hat[0], theta_hat[1], theta_hat[2], theta_hat[3])
		# store cdf points and probability masses for each individual
		print("Version "+str(version[v]) +": Storing CDF values for respondent "+str(i_1)+" out of "+str(obs[v]))
		for p in range(n_points):
			df_cdfs = df_cdfs.append({'id': np.amax(temp_df.id), 'point':np.round(epi[p], 3), 'prob_mass': np.round(cdf_i[p], 3), 'version': version[v] }, ignore_index = True) 
	# save individual bins cdf data
df_cdfs[['id', 'point', 'prob_mass', 'version']].to_csv('v1_v2_beta_cdfs_40.csv')

#--------------------------------------------------------------------------------------------------
#			5.) Fit 2-Bins Data to Triangular Distribution, Split by Endpoint Cases
#--------------------------------------------------------------------------------------------------
df = pd.read_stata('v1_pdfs_cases.dta', columns = ["id","type", "case", "bin_midpoint", "epi_12m_", "v1_bin_min_i", "v1_bin_max_i"]) 
df = df[df.type == 2] # type 2 since fitting to triangular distr.

# individual data is divided into three cases (See stata code): 
# 1. only right endpoint known
# 2. only left endpoint known 
# 3. both endpoints known

# case 1: θ = (l)
df_1 = df[df.case == 1] 
df_1.insert(1, "id2", df_1.groupby(['id']).ngroup())

# case 2: θ = (s)
df_2 = df[df.case == 2] 
df_2.insert(1, "id2", df_2.groupby(['id']).ngroup())

# case 3: θ = ({})
df_3 = df[df.case == 3]
df_3.insert(1, "id2", df_3.groupby(['id']).ngroup())

#  create empty dataframe to store theta estimates
df_triang_bins = pd.DataFrame()
df_triang_bins['id'] = []
df_triang_bins['case'] = []
df_triang_bins['location'] = []
df_triang_bins['scale'] = []
df_triang_bins['mean'] = []
df_triang_bins['median'] = []
df_triang_bins['std'] = []

cases = [df_1, df_2, df_3]

# initial guess for theta by case:
theta0_cases= [[-16],[16],[]]
shape = .5 # this is the assumed value for the parameter c. (Implies isosceles distr for all.)
for c in range(len(cases)):
	#store number of unique ids to iterate over
	cases[c] = cases[c].sort_values(by = ["id2", "bin_midpoint"])
	case = np.amax(cases[c].case).astype('int')
	obs = np.amax(cases[c].id2) + 1
	for i in range(obs):
		temp_df = cases[c][cases[c].id2 == i] 
		r = np.amax(temp_df.v1_bin_max_i) # only relevant for cases 1,3
		l = np.amax(temp_df.v1_bin_min_i) # only relevant for cases 2,3
		c_1 = c+1
		i_1 = i+1 
		print("Bins Case "+str(c_1)+": Triangular distr. fitting for respondent "+str(i_1)+" out of "+str(obs))
		ys = np.asarray(temp_df.epi_12m_) #probabilities 
		ts = np.asarray(temp_df.bin_midpoint) #point values
		theta0 = theta0_cases[c]
		if case == 1:
			y_distr = y_triang_1
		if case == 2:
			y_distr = y_triang_2
		if case == 3:
			y_distr = y_triang_3
		# fit data to beta distribution using non-linear least squares
		if case != 3:
			res = least_squares(fun, theta0, method = 'trf', bounds = [-100, np.inf])
		# calculate implied mean, median, and uncertainty of inflation expectations at the individual level
		if case == 1: #estimate location, not scale
			theta_hat_i = [shape, res.x[0], (r - res.x[0])]
		if case == 2: #estimate scale, not location
			theta_hat_i = [shape, l, (res.x[0])]
		if case == 3: #no estimation required
			theta_hat_i = [shape, l, ( r - l)]
		# generate individual-level summary stats, parameter estimates. Store values.
		sum_stats_i = triang_sum_stats(theta_hat_i)
		df_triang_bins = df_triang_bins.append({'id' : np.amax(temp_df.id), 'case' : np.amax(temp_df.case),'shape': theta_hat_i[0] , 'location' :theta_hat_i[1], 'scale': theta_hat_i[2], 'mean': sum_stats_i[0], 'median':sum_stats_i[1], 'std':sum_stats_i[2]}, ignore_index = True)
df_triang_bins.insert(1, "id2", df_triang_bins.groupby(['id']).ngroup())
df_triang_bins.to_csv('v1_triang_fit.csv')

#--------------------------------------------------------------------------------------------------
#			6.) Fit 2 -Scenarios Data to Triangular Distribution, Split by Endpoint Cases
#--------------------------------------------------------------------------------------------------

df_scen = pd.read_stata('v2_pdfs_cases.dta')
df_scen = df_scen[df_scen.type == 2]

df_1 = df_scen[df_scen.case == 1]
df_1.insert(1, "id2", df_1.groupby(['id']).ngroup())

df_2 = df_scen[df_scen.case == 2]
df_2.insert(1, "id2", df_2.groupby(['id']).ngroup())

df_3 = df_scen[df_scen.case == 3]
df_3.insert(1, "id2", df_3.groupby(['id']).ngroup())

# create empty dataframe to store parameter estimates
df_triang_scen = pd.DataFrame()
df_triang_scen['id'] = []
df_triang_scen['case'] = []
df_triang_scen['location'] = []
df_triang_scen['scale'] = []
df_triang_scen['mean'] = []
df_triang_scen['median'] = []
df_triang_scen['std'] = []

cases = [df_1, df_2, df_3]
shape = .5

for c in range(len(cases)):
	#store number of unique ids to iterate over
	cases[c] = cases[c].sort_values(by = ["id2", "point"])
	case = np.amax(cases[c].case).astype('int')
	obs = np.amax(cases[c].id2) + 1
	for i in range(obs):
		temp_df = cases[c][cases[c].id2 ==i] 
		r = np.amax(temp_df.point) # relevant for cases 1,3
		l = np.amin(temp_df.point) # relevant for cases 2,3
		theta0_cases = [[l],[r-l],[]] 
		c_1 = c+1
		i_1 = i+1
		print("Scenarios Case "+str(c_1)+": Triangular distr. fitting for respondent "+str(i_1)+" out of "+str(obs)+",ID: "+str(np.amax(temp_df.id)))
		ys = np.asarray(temp_df.prob) #probabilities 
		ts = np.asarray(temp_df.point) #point values
		theta0 = theta0_cases[c]
		if case == 1:
			y_distr = y_triang_1
		if case == 2:
			y_distr = y_triang_2
		if case == 3:
			y_distr = y_triang_3
		# fit data to beta distribution using non-linear least squares
		if case != 3:
			res = least_squares(fun, theta0, method = 'trf', bounds = [-100, 200])
		# calculate implied mean, median, and uncertainty of inflation expectations at the individual level
		if case == 1: #estimate location, not scale
			theta_hat_i = [shape, res.x[0], (r - res.x[0])]
		if case == 2: #estimate scale, not location
			theta_hat_i = [shape, l, (res.x[0])]
		if case == 3: #no estimation required
			theta_hat_i = [shape, l, (r - l)]
		# generate individual-level summary stats, parameter estimates. Store values.
		sum_stats_i = triang_sum_stats(theta_hat_i)
		df_triang_scen = df_triang_scen.append({'id' : np.amax(temp_df.id), 'case' : np.amax(temp_df.case),'shape':theta_hat_i[0] , 'location' :theta_hat_i[1], 'scale':theta_hat_i[2], 'mean': sum_stats_i[0], 'median':sum_stats_i[1], 'std':sum_stats_i[2]}, ignore_index = True)
df_triang_scen.insert(1, "id2", df_triang_scen.groupby(['id']).ngroup())
df_triang_scen.to_csv('v2_triang_fit.csv')

#--------------------------------------------------------------------------------------------------
#			7.) Construct and store individual triang. distr.-implied  CDFs.
#--------------------------------------------------------------------------------------------------
grid_min = -20
grid_max = 40
n_points = grid_max - grid_min + 1
epi  = np.linspace(grid_min, grid_max, n_points)

# Construct bins and scenarios beta-implied CDFs
# create empty dataframe to store CDF values 
df_cdfs = pd.DataFrame()
df_cdfs['id'] = []
df_cdfs['point'] = []
df_cdfs['prob_mass'] = []
df_cdfs['version'] = []

version = [1,2]
triang_data = [df_triang_bins, df_triang_scen]
obs = [np.amax(df_triang_bins.id2), np.amax(df_triang_scen.id2)]

for v in range(len(version)): 
	obs_v = obs[v] + 1
	for i in range(obs_v):
		i_1 = i+1
		temp_df = triang_data[v][triang_data[v].id2 == i]
		theta_hat = np.asarray(temp_df[['shape', 'location', 'scale']])[0]
		cdf_i = triang.cdf(epi, theta_hat[0], theta_hat[1], theta_hat[2])
		# store cdf points and probability masses for each individual
		print("Version "+str(version[v]) +": Storing CDF values for respondent "+str(i_1)+" out of "+str(obs_v))
		for p in range(n_points):
			df_cdfs = df_cdfs.append({'id': np.amax(temp_df.id), 'point':np.round(epi[p], 3), 'prob_mass': np.round(cdf_i[p],3), 'version': version[v] }, ignore_index = True) 
	# save individual bins cdf data
df_cdfs[['id', 'point', 'prob_mass', 'version']].to_csv('v1_v2_triang_cdfs_40.csv')

#--------------------------------------------------------------------------------------------------
#							8.) Fit 1-bin responses to uniform distribution
#--------------------------------------------------------------------------------------------------
df = pd.read_stata('v1_pdfs_cases.dta')
df = df[df.type == 1]
df.insert(1, "id2", df.groupby(['id']).ngroup())

# create empty dataframe to store parameter estimates
df_unif_bins = pd.DataFrame()
df_unif_bins['id'] = []
df_unif_bins['case'] = []
df_unif_bins['location'] = []
df_unif_bins['scale'] = []
df_unif_bins['mean'] = []
df_unif_bins['median'] = []
df_unif_bins['std'] = []

obs = np.max(df.id2) + 1
for i in range(obs):
	temp_df = df[df.id2 ==i]
	i_1 = i+1 
	print("Version 1: Unif. distr. fitting for respondent "+str(i_1)+" out of "+str(obs))
	ys = np.asarray(temp_df.epi_12m_) #probabilities 
	ts = np.asarray(temp_df.bin_midpoint) #point values
	r = np.amax(temp_df.v1_bin_max_i)
	l = np.amax(temp_df.v1_bin_min_i)
	y_distr = y_unif_3
	theta_hat_i = [l, r-l]
	# generate individual-level sum stats. Store values
	sum_stats_i = unif_sum_stats(theta_hat_i)
	df_unif_bins = df_unif_bins.append({'id' : np.amax(temp_df.id), 'case' : np.amax(temp_df.case), 'location' : l, 'scale':(r - l), 'mean': sum_stats_i[0], 'median':sum_stats_i[1], 'std':sum_stats_i[2]}, ignore_index = True)
df_unif_bins.insert(1, "id2", df_unif_bins.groupby(['id']).ngroup())
df_unif_bins.to_csv('v1_unif_fit.csv')
#--------------------------------------------------------------------------------------------------
#							9.) Fit 1 scenario responses to uniform distribution
#--------------------------------------------------------------------------------------------------
df = pd.read_stata('v2_pdfs_cases.dta')
df = df[df.type == 1]
df.insert(1, "id2", df.groupby(['id']).ngroup())

# create empty dataframe to store parameter estimates
df_unif_scen = pd.DataFrame()
df_unif_scen['id'] = []
df_unif_scen['case'] = []
df_unif_scen['location'] = []
df_unif_scen['scale'] = []
df_unif_scen['mean'] = []
df_unif_scen['median'] = []
df_unif_scen['std'] = []

delta = .001

obs = np.max(df.id2) + 1
for i in range(obs):
	temp_df = df[df.id2 ==i]
	i_1 = i+1 
	print("Scen: Unif. distr. fitting for respondent "+str(i_1)+" out of "+str(obs))
	ys = np.asarray(temp_df.prob) #probabilities 
	ts = np.asarray(temp_df.point) #point values
	r = ts[0] + delta
	l = ts[0] - delta
	y_distr = y_unif_3
	theta_hat_i = [l, r - l]
	# generate individual-level sum stats. Store values
	sum_stats_i = unif_sum_stats(theta_hat_i)
	df_unif_scen = df_unif_scen.append({'id' : np.amax(temp_df.id), 'case' : np.amax(temp_df.case), 'location' : l, 'scale':(r - l), 'mean': sum_stats_i[0], 'median':sum_stats_i[1], 'std':sum_stats_i[2]}, ignore_index = True)
df_unif_scen.insert(1, "id2", df_unif_scen.groupby(['id']).ngroup())
df_unif_scen.to_csv('v2_unif_fit.csv')

#--------------------------------------------------------------------------------------------------
#			10.) Construct and store individual uniform. distr.-implied  CDFs.
#--------------------------------------------------------------------------------------------------
grid_min = -20
grid_max = 40
n_points = grid_max - grid_min+1
epi  = np.linspace(grid_min, grid_max, n_points)

# Construct bins and scenarios beta-implied CDFs
# create empty dataframe to store CDF values 
df_cdfs = pd.DataFrame()
df_cdfs['id'] = []
df_cdfs['point'] = []
df_cdfs['prob_mass'] = []
df_cdfs['version'] = []

version = [1,2]
unif_data = [df_unif_bins, df_unif_scen]
obs = [np.amax(df_unif_bins.id2), np.amax(df_unif_scen.id2)] 
for v in range(len(version)): 
	obs_v = obs[v] + 1
	for i in range(obs_v):
		i_1 = i+1
		temp_df = unif_data[v][unif_data[v].id2 == i]
		theta_hat = np.asarray(temp_df[['location', 'scale']])[0]
		cdf_i = uniform.cdf(epi, theta_hat[0], theta_hat[1])
		# store cdf points and probability masses for each individual
		print("Version "+str(version[v]) +": Storing CDF values for respondent "+str(i_1)+" out of "+str(obs_v))
		for p in range(n_points):
			df_cdfs = df_cdfs.append({'id': np.amax(temp_df.id), 'point':np.round(epi[p], 3), 'prob_mass': np.round(cdf_i[p],3), 'version': version[v] }, ignore_index = True) 
	# save individual bins cdf data
df_cdfs[['id', 'point', 'prob_mass', 'version']].to_csv('v1_v2_unif_cdfs_40.csv')
