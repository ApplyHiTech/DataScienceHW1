
# coding: utf-8

# In[3]:

import findspark
import pyspark
import numpy as np
import pyspark.sql.functions as sqlFunctions
import matplotlib.pyplot as plt

from operator import add
from pyspark.sql import SQLContext

get_ipython().magic(u'matplotlib inline')


# In[4]:

findspark.init()
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)


# In[5]:

# Full original training set
raw_full_train = sc.textFile("dac/train.txt")

# Final test set split
raw_final_test = sc.textFile("dac/split/test.txt")  # Do not touch during training

# Training set splits
raw_test_3m = sc.textFile("dac/split/test_3m.txt")
raw_train_5m = sc.textFile("dac/split/train_5m.txt")
raw_validation_2m = sc.textFile("dac/split/train_5m.txt")

# Debug set
raw_small_train = sc.textFile("dac/small-train.txt")


# In[6]:

def convert_value(index, value):
    if index < 14:
        return int(value) if value else None
    else:
        return value if value else None

def convert_line(line):
    return [convert_value(i, value) for i, value in enumerate(line.split("\t"))]


# In[7]:

# Change the data types of the datasets so that the RDD's include Int's and Strings. 
full_train = raw_full_train.map(convert_line)
final_test = raw_final_test.map(convert_line)
test_3m = raw_test_3m.map(convert_line)
train_5m = raw_train_5m.map(convert_line)
validation_2m = raw_validation_2m.map(convert_line)

#debug = raw_small_train.map(convert_line)


# In[9]:

def int_column_histogram(col_num, col,numb_bins=10):
    bins, counts = col.histogram(numb_bins)
    total = sum(counts)    
    print "Column %d histogram\n\tBins=%s\n\tCounts=%s (total=%d)" % (col_num, bins, counts, total)
    # TODO: display graph of histogram
    # TODO: better buckets for histogram (smart sub-dividing)
        #sum the counts
        #max of the counts
        #if  > 25%
    return bins,counts

def int_columns_histograms(data):
    bins=[]; counts=[]
    for i, col in enumerate(column_iter(data)):
        col_num = i + 1
        if is_integer_col_num(col_num):
            bins1,counts1 = int_column_histogram(col_num, col)
            bins.append(bins1) #bin values
            counts.append(counts1) #count inside bins
            
    return bins,counts

def bin_range_labels(bins):
    #Nicely display these.
    "{:0,.3f} - {:0,.0f}"
    return ["%s---%s" % ("{:.1E}".format((bins[i])),"{:.2E}".format((bins[i+1]))) for i in range(len(bins) - 1)]




# In[44]:

def get_column_num(data, col_num):
    return data.map(lambda row: row[col_num])

def column_filter_null(column):
    return column.filter(lambda row: row is not None)

def column_count(data):
    return len(data.take(1)[0])

def is_integer_col_num(col_num):
    return col_num > 1 and col_num < 15

def is_label_col_num(col_num):
    return col_num == 1

def is_categorical_col_num(col_num):
    return col_num >= 15

def column_iter(data):
    for i in range(column_count(data)):
         yield get_column_num(data, i)


# In[48]:

def cat_columns_histogram(data,numb_bins=10):
    hashes=[]; counts_all=[]; remainder=[];
    
    for i, col in enumerate(column_iter(data)):
        col_num = i + 1
        if is_categorical_col_num(col_num):
            key_counts = col.map(lambda key: (key, 1)).reduceByKey(add)
            sorted_counts = sorted(key_counts.collect(), key=lambda t: t[1], reverse=True)
            labels = [v[0] for v in sorted_counts]
            counts = [v[1] for v in sorted_counts]
            print i
            print col
            hashes.append(labels[:numb_bins]) #bin values
            counts_all.append(counts[:numb_bins]) #count inside bins
            remainder.append(sum(counts[numb_bins:]))
            
    return hashes,counts_all,remainder


# In[22]:

# This represents the Histograms for Features. 
# For Integer Feature we do not compute "Other Values"
# 
# x_values = 1D array of the x_values {bins OR category names}
# y_values = 1D array of the y_values {counts of uniques in bins or category name}
# isCategory = True if feature is category. 
#            = False if feature is integer.
# Z_other_values = sum of the counts of the remaining categories for category feature.
#
#
def disp_Histogram(x_values,y_values,isCategory, column_numb,z_other_value=0):
        range_x = min(len(x_values),10)
        x1 = np.arange(range_x)
        x2 = []
        
        
        # Category Feature
        if (isCategory):
            type_of_feature = "Category"
            x_label = "Category as a Hashed value"
            #Add Other column
            for i in x_values: 
                x2.append(str(i))
            
            x1 = np.append(x1,10)
            x2.append('Other')
            #print "HI",z_other_value
            y_values.append(z_other_value)

        # Integer Feature
        else: 
            type_of_feature = "Integer"
            x_label = "Bins of Integer values"
            x2 = bin_range_labels(x_values)

        
        plt.title('%s Feature %s Histogram' % (type_of_feature,column_numb))
        plt.ylabel('Count of values')
        plt.xlabel('The %s' % x_label)
        print ("X: %s, Y: %s" % (x1,y_values))
        plt.xticks(x1, x2,rotation=45)
        print "X: %s, Y: %s" % (len(x1),len(y_values))
        plt.bar(x1, y_values,log=True)
        plt.show()

    


# In[23]:

# The following shows all the histograms for all of the features. 
def show_all_histograms(x,y,featureType,z_other_value=0):
    isCategory = True
     
    if featureType=="Integer":
        isCategory = False
       
        for i in range(len(x)):
            disp_Histogram(x[i],y[i],isCategory,i,z_other_value=0)
    else:
        isCategory = True
        for i in range(len(x)):
            #print z_other_value[i]
            disp_Histogram(x[i],y[i],isCategory,i,z_other_value[i])




# ## This computes the Mean, StDev, Kurtosis, and Skewness for each Integer Feature. 
# 
# It outputs the results for 13 features.

# In[40]:

def print_column_summary_details(col_num, mean,std,kurtosis, skewness):
    print("Column #%2d: mean=%-10.3f std=%-10.3f Kurtosis=%-10.3f Skewness=%-10.3f" % (col_num, mean,std, kurtosis, skewness))

def int_columns_detail_stats(data):
    df = sqlContext.createDataFrame(data)
    for i, col in enumerate(column_iter(data)):
        col_num = i + 1
        if is_integer_col_num(col_num):            
            col = df["_%s" % col_num]
            m_col = sqlFunctions.mean(col)
            m_results = df.select(m_col.alias("mean")).collect()[0]
            std_col = sqlFunctions.stddev(col)
            std_results = df.select(std_col.alias("stddev")).collect()[0]
            k_col = sqlFunctions.kurtosis(col)
            k_result = df.select(k_col.alias("kurtosis")).collect()[0]
            s_col = sqlFunctions.skewness(col)
            s_result = df.select(s_col.alias("skewness")).collect()[0]
            
            print_column_summary_details(col_num, m_results.mean,std_results.stddev,k_result.kurtosis, s_result.skewness)


# ### The following functions compute the Summary Statistics

# In[31]:

#This code computes the histograms based on the training sets.
#The code computes 10 bins by default.

x_int_val, y_int_val =int_columns_histograms(train_5m)
show_all_histograms(x_int_val,y_int_val,"Integer")


# In[43]:

#Compute and display Integer Summary Statistics, Mean, Stdev, Skewness, Kurtosis)
int_columns_detail_stats(train_5m)


# In[ ]:

# Compute and display Category Histograms
hashes, counts, remainder =cat_columns_histogram(train_5m)
show_all_histograms(hashes, counts,"Category",remainder)


# In[ ]:



