import numpy as np
import h5py
# The h5py package is required to read the MacDonald database.

import PyMultipolator0420 as PM



def return_albedo(alb_database, log_m, log_g, T, f, H2O = True):
	''' Quick function to open albedo database for a specified metallicity,
		gravity, effective temperature, and sedimentation efficiency then write
		output to a .txt file.
	'''
	# Check that given parameter combination is valid
	if ((log_m not in np.array([0.0, 0.5, 1.0, 1.5, 1.7, 2.0])) or
		(log_g not in np.around(np.linspace(2.0, 4.0, 21), decimals=1)) or
		(T not in np.linspace(150, 400, 26).astype(np.int64)) or
		(f not in np.linspace(1, 10, 10).astype(np.int64))):
		raise Exception("Specified parameter combination not included in albedo database.")
	
	# Extract wavelength array
	wl = np.array(alb_database['wl'])
	
	# Navigate to specified parameter combination
	data = alb_database['m{:.1f}_g{:.1f}_T{:d}_f{:d}'.format(log_m, log_g, int(T), f)]
	
	# Extract albedo with or without H2O opacity
	
	if H2O == True:
		alb = np.array([wl, data['Albedo']])
	
	elif H2O == False:
		alb = np.array([wl, data['Albedo_no_H2O']])
	
	else: # Exception handling
		raise Exception
	
	# Return the albedo
	return alb


def dataFetcherHDF5(modelLocation):
	'''
	Purpose:
		This function is a data fetcher, designed to grab data from an HDF5 binary file.
	Input:
		modelLocation	:	Index values corresponding to a model. Each index indicates a value of a different parameter.
	Output: 
		result			:	A two-dimesional model loaded from the HDF5 file source.
	'''

	# We first translate the index values into parameter values.
	ParameterValues = [Xknown[i][modelLocation[i]] for i in range(len(modelLocation))]

	# We then call return_albedo() to retrieve the model.
	loadModel = return_albedo(alb_database, *ParameterValues)

	# Extract the wlgrid (x) and ymeas (y) arrays.
	wlgrid = loadModel[0]
	ymeas = loadModel[-1]

	# Combine them into result and return.
	result = [wlgrid, ymeas]
	return result


# First I must load in the h5py file that holds the models.
alb_database = h5py.File('./Cool_giant_albedo_database.hdf5', 'r')

# PyMultipolator needs to know what values of each parameter are valid; thus we generate this list of lists.
Xknown = [ [0.0, 0.5, 1.0, 1.5, 1.7, 2.0], np.around(np.linspace(2.0, 4.0, 21), decimals=1),
			 np.linspace(150, 400, 26).astype(np.int64), np.linspace(1, 10, 10).astype(np.int64) ]

# Picking an arbitrary set of parameters to calculate an albedo for.
Xarray = [ 0.25, 2.44, 306, 4.3 ]

# Run PyMultipolator.
# Note: We need to pass PyMultipolator three objects. We must tell it the "known" parameter values values, we must tell it the array of
# parameter values we want to approximate, and we must pass it a function which can be passed index values and will return an appropriate
# albedo.
final = PM.SingleCallMP(Xknown,Xarray,dataFetcherHDF5)

# Print our output.
print(final)