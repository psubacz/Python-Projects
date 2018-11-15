import numpy as np

training_data = np.load('Data\\training_data_200x150_2.npy')
print('Finished loading data 1')

loaded_final_data = np.load('Data\\training_data_200x150_4.npy')
print('Finished loading data 2')

final_data = np.concatenate((loaded_final_data,training_data))
print('Finish Concatenting, Saving Data')

#np.save('Data\\Final_Training_Data_200x150_x.npy',final_data)
#print('Data Saved')