# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:15:57 2018
Modified by Umut KARACALARLI
Modifications:
    - PCA replaced with Fisher Score for feeature reduction
    - Feature reduction is done for multiple feature sizes
    - SVM kernels poly, sigmoid and rbf added
    - Attributes vectorized manually
    - Confusion matrix added
    - Output directed to file
    - Elapsed time calculated by module (FS, Normalization and SVM)


@Original author: Gökberk GÜLGÜN (pwnageco) - https://github.com/pwnageco

https://github.com/ukaracalarli/Fisher-Score-and-SVM is licensed under the GNU General Public License v3.0

"""
# In[10]:
# -*- coding: utf-8 -*-
from __future__ import division
import os
import pandas as pd
import urllib
from skfeature.function.similarity_based import fisher_score
from sklearn import svm, metrics, preprocessing
from sklearn.metrics import confusion_matrix
import time
import datetime
import collections
import sys


# In[20]:
class IDS():

    def __init__(self,num_fs):
        
        """ Initalization of dataset configure predefined columns            
        """
        # env variable is set to 0 to read the files on linux machine and set to 1 to read the files on Windows machine.
        env=1
        if env==1:
            os.chdir('C:\Users\ukara\PycharmProjects\ids_svm1')
            self.train_data_from_text = urllib.urlopen('.\/data\/kddcup.data_10_percent_corrected')
            self.test_data_from_text = urllib.urlopen('.\/data\/corrected')
        else:
            os.chdir('/home/ukaraca/ids')
            self.train_data_from_text = urllib.urlopen('/home/ukaraca/ids/data/kddcup.data_10_percent_corrected')
            self.test_data_from_text = urllib.urlopen('/home/ukaraca/ids/data/corrected')
            
        """ Train data read from frame """
        self.class_train = pd.read_csv(self.train_data_from_text, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
        self.class_test = pd.read_csv(self.test_data_from_text, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
        
        
        #row_count is used to do the test with complete or partial data
        self.row_count = int(len(self.class_train) * 1)
        
        #num_fea variable defines the number of features to be used in SVM as output of Fisher Score feature reduction
        #num_fea is given as parameter with num_fs
        self.num_fea = num_fs
        self.features = []
        #score variable holds accuracy score of SVM
        self.score = 0.0

        #data size is defined with row count 
        self.class_train = self.class_train[:self.row_count]
        self.class_test = self.class_test[:self.row_count]
       
        #data frames are initialized and NaN fields are filled with 0
        self.class_train_reduced = pd.DataFrame()
        self.class_train = self.class_train.fillna(0)
        self.class_test_reduced = pd.DataFrame()
        self.class_test = self.class_test.fillna(0)
                 
    
        """ Change of train classes by tagging normal or attack """
        self.class_train.loc[(self.class_train['Class'] !='normal.'),'Class'] = 'attack'
        self.class_train.loc[(self.class_train['Class'] =='normal.'),'Class'] = 'normal'
        """ Change of test classes of tagging normal or attack """
        self.class_test.loc[(self.class_test['Class'] !='normal.'),'Class'] = 'attack'
        self.class_test.loc[(self.class_test['Class'] =='normal.'),'Class'] = 'normal'
       
        #list and dictionaries to be used in ordering the features by importance
        self.list = []
        self.dic = {}
        self.odic= {}
       
        #non numeric protocol types are vectorized
        self.protocol_dict = {'tcp':1, 'udp':2, 'icmp':3}
        self.class_train['protocol_type'].replace(self.protocol_dict, inplace=True)
        self.class_test['protocol_type'].replace(self.protocol_dict, inplace=True)
        
        #non numeric service types are vectorized
        self.service_dict ={'http':1,
                       'ecr_i':2,
                       'smtp':3,
                       'domain_u':4,
                       'ftp_data':5,
                       'private':6,
                       'eco_i':7,
                       'finger':8,
                       'ftp':9,
                       'ntp_u':10,
                       'telnet':11,
                       'auth':12,
                       'pop_3':13,
                       'other':14,
                       'time ':15,
                       'domain':16, 
                       'rje':17,
                       'gopher':18, 
                       'ssh':19,
                       'mtp':20,
                       'login':21,
                       'link':22,
                       'nntp':23,
                       'name':24,
                       'imap4':25,
                       'whois':26,
                       'remote_job':27,
                       'daytime':28,
                       'ctf':29,
                       'time':30,
                       'X11':31,
                       'IRC':32,
                       'tim_i':33,
                       'http_443':34,
                       'systat':35,
                       'ldap':36,
                       'printer':37,
                       'sunrpc':38,
                       'urp_i':39,
                       'Z39_50':40,
                       'bgp':41,
                       'vmnet':42,
                       'sql_net':43,
                       'netbios_dgm':44,
                       'netbios_ssn':45,
                       'netbios_ns':46,
                       'uucp_path':47,
                       'pop_2':48,
                       'csnet_ns':49,
                       'iso_tsap':50,
                       'hostnames':51,
                       'supdup':51,
                       'netstat':52,
                       'discard':53,
                       'echo':54,
                       'klogin':55,
                       'kshell':56,
                       'uucp':57,
                       'courier':58,
                       'efs':59,
                       'shell':60,
                       'exec':61,
                       'nnsp':62,
                       'red_i':63,
                       'tftp_u':64,
                       'pm_dump':65,
                       'urh_i':66,
                       'icmp':67}
        
        self.class_train['Service'].replace(self.service_dict, inplace=True)
        self.class_test['Service'].replace(self.service_dict, inplace=True)
        
        #non numeric flag types are vectorized
        self.flag_dict ={'SF':1,
                    'REJ':2,
                    'SH':3,
                    'RSTR':4,
                    'RSTO':5,
                    'S1':6,
                    'S0':7,
                    'RSTOS0':8,
                    'S2':9,
                    'S3':10,
                    'OTH':11}
        
        self.class_train['Flag'].replace(self.flag_dict, inplace=True)
        self.class_test['Flag'].replace(self.flag_dict, inplace=True)
        
        #non numeric class types are vectorized
        self.class_dict ={'normal':0, 'attack':1}
        self.class_train['Class'].replace(self.class_dict, inplace=True)
        self.class_test['Class'].replace(self.class_dict, inplace=True)

    """ Train and test data feature reduction"""
    def fisher_feature_reduction(self,down,up) :
        
        #importance of attributes are measured and listed
        score = fisher_score.fisher_score(self.class_train.values[down:up,:-1], self.class_train.iloc[down:up,-1])

        #attributes are saved to dictionary with their importance value in cumulative way
        self.to_dict(score)

        return score
    
    
    """ Normalizing dataset  """
    def normalizing_datasets(self):
        
        #data is normalized before processed in SVM
        standard_scaler = preprocessing.StandardScaler()
        #features are ordered by importance
        self.to_list()
        #kaydirma
        self.features= self.list[0:self.num_fea]
        print "features used in svm: ", self.features
        
        
        #only most important features of size defined by num_fea are selected
        self.class_train_reduced  = (self.class_train.iloc[:, self.features])
        self.class_test_reduced = (self.class_test.iloc[:, self.features])
        
        #data with selected features are normalized and converted to data frame
        self.class_train_scaled = standard_scaler.fit_transform(self.class_train_reduced.values)
        self.class_train_normalized = pd.DataFrame(self.class_train_scaled)
        self.class_test_scaled = standard_scaler.fit_transform(self.class_test_reduced.values)
        self.class_test_normalized = pd.DataFrame(self.class_test_scaled)            
  
 
    """ SVM with third party tools """
    def svm_with_third_party(self,kernel):
        
        # SVM is initialized with given kernel as parameter
        svm_object = svm.SVC(kernel=kernel, max_iter=100000000, C=2, gamma=0.02, decision_function_shape='ovo')
                
        #SVM is trained with normalized training data
        svm_object.fit(self.class_train_normalized, self.class_train.values[:,-1])
        
        #SVM is tested wirh normalized test data
        svm_predict = svm_object.predict(self.class_test_normalized)
        
        #Accuracy score of SVM is calculated
        self.score = svm_object.score(self.class_test_normalized, self.class_test.values[:,-1])
        print 'Accuracy Score: ', self.score
        
        #metrics of SVM result is printed to output
        print 'SVM metrics: ', metrics.classification_report(self.class_test.values[:,-1], svm_predict)
        
        #Confusion matrix (CM) is calculated
        tn, fp, fn, tp = confusion_matrix(self.class_test.values[:,-1], svm_predict).ravel()
        
        #Percentage of each CM value is calculated
        size = self.class_test.shape[0]
        print ('True Positive: ',tp,'% ',tp*100/size)
        print ('True Negative: ',tn, '% ',tn*100/size)
        print ('False Positive: ',fp, '% ',fp*100/size)
        print ('False Negative: ',fn, '% ',fn*100/size)
        
    def to_dict(self,myList):
        
        #Attributes are saved to dictionary cumulatively
        counter = 0
        for items in myList:
            if counter in self.dic.keys():
                self.dic[counter] += myList[counter]
            else:
                self.dic[counter] = myList[counter]
            counter += 1
    
    def to_list(self):
        
    #Attributes are ordered by importance value and converted to list
        self.odic = collections.OrderedDict(sorted(self.dic.items(), key=lambda t: t[1], reverse=True))
        self.list = self.odic.keys()
        print self.odic
        #self.list = list(reversed(self.list))
        		
      
# In[30]: 

#Output is directed to file
orig_stdout = sys.stdout

file_name = 'results_fs_svm_'
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
file_name = file_name + timestamp + '.txt'
print "first file name: ", file_name    
f = open(file_name, 'w')

start_time = time.time()
print('start time: ',
    datetime.datetime.fromtimestamp(
        start_time).strftime('%Y-%m-%d %H:%M:%S'))
    
sys.stdout = f

# In[40]:
#Start time is saved
start_time = time.time()

#Size of features to be used are defined
#kaydirma
fss = [41]


#Code is run for every size in the list
for num_fs in fss:    
    ids = IDS(num_fs)
    #print 'Her turda degistirilecek ozellik sayisi: ', ids.features

    # In[50]:
    # Feature reduction using Fisher Score
    #Start time of Fisher Score method is saved
    fisher_start_time = time.time()
    
    #Fisher Score is fed with data in chunks of 5000 rows to fit it in memory
    chunk_size = range(0,len(ids.class_train),5000)
    i = 1
    while i < len(chunk_size):
        ids.to_dict(ids.fisher_feature_reduction(chunk_size[i-1],chunk_size[i]))
        i += 1
    
    #Fisher Score is fed with remainder data chunk, from division of data size into chunk size
    #e.g. remainder chunk of 12000/5000 is 2000 rows.
    ids.to_dict(ids.fisher_feature_reduction(chunk_size[-1],len(ids.class_train)))
    
    #End time of fisher score method is saved
    fisher_end_time = time.time()
    
    #Elapsed time of fisher score is saved and printed
    fisher_elapsed_time = fisher_end_time - fisher_start_time
    print('Fisher elapsed time: ', int(fisher_elapsed_time))
    
    	# In[51]: 
     
    ids.to_list()
    
    	# In[52]: 
    ids.list
    
    	# In[53]: 
    
    list(ids.class_train.iloc[:,ids.list])
    # In[60]:
        # Data is normalized by using Standard Scaler
        #Start time of normalization is saved
    norm_start_time = time.time()
    
    #Data is normalized before SVM
    ids.normalizing_datasets()
       
    #End time of normalization is saved
    norm_end_time = time.time()
    
    #Elapsed time of normalization is saved
    norm_elapsed_time = norm_end_time - norm_start_time
    print('Normalization elapsed time: ', int(norm_elapsed_time))
        
    
    	# In[70]:
    # Data is classiffied by using SVM
    
    #Kernels in the list to be used in SVM
    	#kernels = ['linear','poly','rbf','sigmoid']
    kernels = ['rbf'] 
    print 'Features in order by value.', ids.list
        
    #For every kernel listed above, SVM is run
    for k in kernels:
    		#Start time of SVM is saved
        svm_start_time = time.time()
        print 'Svm is run with kernel: ', k
        #SVM is run with the kernel given as parameter 
        ids.svm_with_third_party(k)
        
        #End time of SVM is saved
        svm_end_time = time.time()
        
        #Elapsed time of SVM is saved and printed
        svm_elapsed_time = svm_end_time - svm_start_time
        print('SVM elapsed time: ', int(svm_elapsed_time))

        
    
# In[80]:

#Stop time of whole code is saved and printed
stop_time = time.time()
print 'Toplam gecen zaman: ', int(stop_time - start_time)

#Output is directed to back screen and outout file is closed
sys.stdout = orig_stdout
f.close()
print "output written to file: ", file_name