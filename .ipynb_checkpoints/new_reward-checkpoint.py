
# coding: utf-8

#Class declaration for ENO class

#INPUT : CSV file designated by location and year

#OUTPUTS: Harvested Energy (henergy)
#         Forecast (fcast)
#         Time (year, day, hr)
#         Place (location)
#         Flags (end_of_day, end_of_year)

#METHODS: To shuffle days randomly (shuffle_days())
#         To emulate days of only a certain daytype (daytype(x))

import pandas as pd
import numpy as np


class ENO(object):
    
    #no. of forecast types is 6 ranging from 0 to 5
  
    def __init__(self, location='tokyo', year=2010, shuffle=False):
        self.location = location
        self.year = year
        self.day = None
        self.hr = None
        
        self.shuffle = shuffle

        self.TIME_STEPS = None #no. of time steps in one episode
        self.NO_OF_DAYS = None #no. of days in one year
        
        self.sradiation = None #matrix with GSR for the entire year
        self.senergy = None #matrix with harvested energy data for the entire year
        self.fforecast = None #matrix with forecast values for each day
        

        self.henergy = None #harvested energy variable
        self.fcast = None #forecast variable
        self.sorted_days = [] #days sorted according to day type
    
    #function to get the solar data for the given location and year and prep it
    def get_data(self):
        #CSV files contain the values of GSR (Global Solar Radiation in MegaJoules per meters squared per hour)
        file = './data/' + self.location +'/' + str(self.year) + '.csv'
        #skiprows=4 to remove unnecessary title texts
        #usecols=4 to read only the Global Solar Radiation (GSR) values
        solar_radiation = pd.read_csv(file, skiprows=4, encoding='shift_jisx0213', usecols=[4])
        
        #convert dataframe to numpy array
        solar_radiation = solar_radiation.values
        #reshape solar_radiation into no_of_daysx24 array
        sradiation = solar_radiation.reshape(-1,24)
        #convert missing data in CSV files to zero
        sradiation[np.isnan(sradiation)] = 0
        if(self.shuffle): #if class instatiation calls for shuffling the day order. Required when learning
            np.random.shuffle(sradiation) 
        self.sradiation = sradiation
        
        
        #GSR values (in MJ/sq.mts per hour) need to be expressed in mW
        # Conversion is accomplished by 
        # solar_energy = GSR(in MJ/m2/hr) * 1e6 * size of solar cell * efficiency of solar cell /(60x60) *1000 (to express in mW)

        self.senergy = self.sradiation * 1e6 * (55e-3 * 70e-3) * 0.15 * 1000/(60*60) 

        return 0
    
    #function to map total day radiation into type of day ranging from 0 to 5
    #the classification into day types is quite arbitrary. There is no solid logic behind this type of classification.
    
    def get_day_state(self,tot_day_radiation):
        if (tot_day_radiation < 3.5):
            day_state = 0
        elif (3.5 <= tot_day_radiation < 7):
            day_state = 1
        elif (7 <= tot_day_radiation < 12):
            day_state = 2
        elif (12 <= tot_day_radiation < 15):
            day_state = 3
        elif (15 <= tot_day_radiation < 17.5):
            day_state = 4
        else:
            day_state = 5
        return int(day_state)
    
    def get_forecast(self):
        #create a perfect forecaster.
        tot_day_radiation = np.sum(self.sradiation, axis=1) #contains total solar radiation for each day
        get_day_state = np.vectorize(self.get_day_state)
        self.fforecast = get_day_state(tot_day_radiation)
        
        #sort days depending on the type of day and shuffle them; maybe required when learning
        for fcast in range(0,6):
            fcast_days = ([i for i,x in enumerate(self.fforecast) if x == fcast])
            np.random.shuffle(fcast_days)
            self.sorted_days.append(fcast_days)
        return 0
    
    def reset(self,day=0): #it is possible to reset to the beginning of a certain day
        
        self.get_data() #first get data for the given year
        self.get_forecast() #calculate the forecast
        
        self.TIME_STEPS = self.senergy.shape[1]
        self.NO_OF_DAYS = self.senergy.shape[0]
        
        self.day = day
        self.hr = 0
        
        self.henergy = self.senergy[self.day][self.hr]
        self.fcast = self.fforecast[self.day]
        
        end_of_day = False
        end_of_year = False
        return [self.henergy, self.fcast, end_of_day, end_of_year]

    
    def step(self):
        end_of_day = False
        end_of_year = False

        if(self.hr < self.TIME_STEPS - 1):
            self.hr += 1
            self.henergy = self.senergy[self.day][self.hr] 
        else:
            if(self.day < self.NO_OF_DAYS -1):
                end_of_day = True
                self.hr = 0
                self.day += 1
                self.henergy = self.senergy[self.day][self.hr] 
                self.fcast = self.fforecast[self.day]
            else:
                end_of_day = True
                end_of_year = True
        
        return [self.henergy, self.fcast, end_of_day, end_of_year]



#Continuous Adaptive Power Manager using default ENO class
class CAPM (object):
    def __init__(self,location='tokyo', year=2010, shuffle=False, trainmode=False):

        #all energy values i.e. BMIN, BMAX, BOPT, HMAX are in mWhr. Assuming one timestep is one hour
        
        self.BMIN = 0.0                #Minimum battery level that is tolerated. Maybe non-zero also
        self.BMAX = 9250.0            #Max Battery Level. May not necessarily be equal to total batter capacity [3.6V x 2500mAh]
        self.BOPT = 0.5 * self.BMAX    #Optimal Battery Level. Assuming 50% of battery is the optimum
        
        self.HMIN = 0      #Minimum energy that can be harvested by the solar panel.
        self.HMAX = 500   #Maximum energy that can be harvested by the solar panel. [500mW]
        
        self.DMAX = 500      #Maximum energy that can be consumed by the node in one time step. [~ 3.6V x 135mA]
        self.N_ACTIONS = 10  #No. of different duty cycles possible
        self.DMIN = self.DMAX/self.N_ACTIONS #Minimum energy that can be consumed by the node in one time step. [~ 3.6V x 15mA]
        
        self.batt = None      #battery variable
        self.enp = None       #enp at end of hr
        self.henergy = None   #harvested energy variable
        self.fcast = None     #forecast variable
        
        self.location = location
        self.year = year
        self.shuffle = shuffle
        self.trainmode = trainmode
        self.eno = ENO(self.location, self.year, shuffle)

        self.no_of_day_state = 6;

  
    
    
    def reset(self,day=0,batt=-1):
        henergy, fcast, day_end, year_end = self.eno.reset(day) #reset the eno environment
        if(batt == -1):
            self.batt = self.BOPT
        else:
            self.batt = batt
            
        self.batt = np.clip(self.batt, self.BMIN, self.BMAX)
        self.enp = self.BOPT - self.batt #enp is calculated
        self.henergy = np.clip(henergy, self.HMIN, self.HMAX) #clip henergy within HMIN and HMAX
        self.fcast = fcast
        
        norm_batt = self.batt/self.BMAX
        norm_enp = self.enp/(self.BMAX/2)
        norm_henergy = self.henergy/self.HMAX
        norm_fcast = self.fcast/(self.no_of_day_state-1)

        c_state = [norm_batt, norm_enp, norm_henergy, norm_fcast] #continuous states
        reward = 0
        
        return [c_state, reward, day_end, year_end]
    
    def getstate(self): #query the present state of the system
        norm_batt = self.batt/self.BMAX
        norm_enp = self.enp/(self.BMAX/2)
        norm_henergy = self.henergy/self.HMAX
        norm_fcast = self.fcast/(self.no_of_day_state-1)
        c_state = [norm_batt, norm_enp, norm_henergy, norm_fcast] #continuous states

        return c_state

    #reward function
    def rewardfn(self):
        R_PARAM = 20000 #chosen empirically for best results
        mu = 0
        sig = 0.05*R_PARAM #knee curve starts at approx. 2000mWhr of deviation
        
        if(np.abs(self.enp) <= 0.12*R_PARAM):
            norm_reward = (np.exp(-np.power((self.enp - mu)/sig, 2.)/2) / np.exp(-np.power((0 - mu)/sig, 2.)/2))
        else:
            norm_reward = -0.25 - 2.5*np.abs(self.enp/R_PARAM)
    
        return norm_reward
        
    
    def step(self, action):
        day_end = False
        year_end = False
        reward = 0
        
        action = np.clip(action, 0, self.N_ACTIONS-1) #action values range from (0 to N_ACTIONS-1)
        e_consumed = (action+1)*self.DMAX/self.N_ACTIONS   #energy consumed by the node
        
        self.batt += (self.henergy - e_consumed)
        self.batt = np.clip(self.batt, self.BMIN, self.BMAX) #clip battery values within permitted level
        
        #code to record all the times battery levels have been exceeded
        self.enp = self.BOPT - self.batt
        
        #proceed to the next time step
        self.henergy, self.fcast, day_end, year_end = self.eno.step()
        self.henergy = np.clip(self.henergy, self.HMIN, self.HMAX) #clip henergy within HMIN and HMAX

        if(day_end): #if eno object flags that the day has ended then give reward
            reward = self.rewardfn()
            
            if (self.trainmode): #reset battery to optimal level if limits are exceeded when training
                if(self.batt == self.BMIN or self.batt == self.BMAX ):
                    self.batt = self.BOPT
                    reward = reward - 2 #penalty for violating battery limits
                
        norm_batt = self.batt/self.BMAX
        norm_enp = self.enp/(self.BMAX/2)
        norm_henergy = self.henergy/self.HMAX
        norm_fcast = self.fcast/5

        c_state = [norm_batt, norm_enp, norm_henergy, norm_fcast] #continuous states
        return [c_state, reward, day_end, year_end]

