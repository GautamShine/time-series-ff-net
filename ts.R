require(forecast) # R. Hyndman's forecast: ARIMA, ETS, AR-NN
require(timeDate) # Handles dates and holidays
require(caret)    # Neural nets

# Helper functions
#####################################################################################

# compute_error
# Returns the average percent deviation in the predicted time series
compute_error <- function(pred, data, start, end){
  diff <- pred - data[start:end]
  error <- 100*mean(abs(diff/data[start:end]))
  return(error)
}

# plot_pred
# Plots the predicted values of the time series (pred) v. the actual (data).
# p1, p2, p3 are model parameters that may go into the plot title.
plot_pred <- function(pred, data, start, end, title, p1_name, p1, p2_name, p2, p3_name, p3){
  n <- end - start + 1
  plot(1:n, data[start:end], type="l", col="red", 
       main=paste(title, p1_name, p1, p2_name, p2, p3_name, p3, sep=" "),
       xlab="Time (Days)", ylab="Sales (Million USD)")
  lines(1:n, pred, type="l", col="blue")
}

# rank_reduce
# Uses singular value decomposition to reduce the rank of the data to the
# number of leading components given by num_comp.
rank_reduce <- function(data, num_comp){
  rr <- svd(data, nu=num_comp, nv=num_comp)
  data <- rr$u %*% diag(rr$d[1:num_comp]) %*% t(rr$v)
  return(data.frame(data))
}

# Main Code
#####################################################################################

data <- read.csv("file.csv", header=TRUE)
end <- length(data[,1]) # index of last data point

ar_vals <- seq(5,15,5) # auto-regression values to try in neural net
hl_vals <- seq(5,30,5) # number of hidden layer nodes to try in neural net
tr_vals <- seq(0.4,0.8,0.2) # percent of data to use for training
ann_error <- array(0, dim=c(length(ar_vals), length(hl_vals), length(tr_vals))) # ANN errors
arima_error <- rep(0,length(tr_vals)) # ARIMA errors for comparison

kc <- 0
for(k in tr_vals){
  kc <- kc+1 # index for training set size
  
  start <- floor(tr_vals[kc]*end) # Use this fraction of samples for training
  num_pred <- end - start + 1 # Number of elements to be predicted
  
  time <- seq(1,start-1)
  rev <- as.numeric(gsub( "[$,]", "", as.character(data[time,2])))/1e6
  time_actual <- seq(1:end)
  rev_actual <- as.numeric(gsub( "[$,]", "", as.character(data[,2])))/1e6
  plot(time_actual, rev_actual, type="l") # plot actual revenue
  
  thanksgiving <- rep(0,end)
  christmas <- rep(0,end)
  year <- rep(0,end)
  
  # assemble dummy variables for special days to serve as regressors
  # this helps catch unusual spikes on those days
  for(i in 1:end){
    date <- as.Date(data[i,1],format="%m/%d/%Y")
    year[i] <- as.numeric(format(date, "%Y"))
    thgv <- holiday(as.numeric(format(date, "%Y")), Holiday="USThanksgivingDay")
    xmas <- holiday(as.numeric(format(date, "%Y")), Holiday="USChristmasDay")
    if(as.numeric(date) == as.numeric(thgv)){
      thanksgiving[i:(i+4)] <- 1
    }
    if(as.numeric(date) == as.numeric(xmas)){
      christmas[(i-10):(i+5)] <- 1
    }
  }
  
  special_days <- cbind(thanksgiving, christmas)
  
  # forecast auto.arima: ARIMA with multi-seasonality using Fourier components
  #####################################################################################
  
  # auto.arima
  # See R. Hyndman's "Forecasting: principles and practice" and forecast package.
  # AR, I, and MA orders are determined using the Akaike information critertion.
  # Fourier components are used for multiple seasonality and dummy vectors
  # are used as regressors for special days.
  x <- msts(rev,seasonal.periods=c(7,365))
  z <- fourier(x, K=c(3,15))
  zf <- fourierf(x, K=c(3,15), h=num_pred)
  fit <- auto.arima(x, xreg=cbind(z,special_days[1:(start-1),]), seasonal=FALSE)
  pred <- forecast(fit, xreg=cbind(zf,special_days[start:end,]), h=num_pred)
  plot_pred(pred$mean, rev_actual, start, end, "ARIMA - ",
            "p:", pred$model$arma[1], "q:", pred$model$arma[2], "d:", pred$model$arma[3])
  error <- compute_error(pred$mean, rev_actual, start, end)
  arima_error[kc] <- error
  print("ARIMA")
  print(paste("Total MSE:", error, sep=" "))
  cat("\n")
  
  # caret avvNNet: Neural net with STL decomposition and multi-seasonality
  #####################################################################################
  
  # ann_iterative
  # Computes time series prediction using a neural net object for any time horizon
  # by iteratively forecasting out one time step and using the prediction as
  # an autoregression input for the next time step.
  ann_iterative <- function(ann, num_ar, num_min, df, df_new, num_pred){
    fc <- matrix(-1, 1, num_pred+num_min)
    for(i in 1:num_min){
      fc[i] <- df[length(df[,1])-num_min-1+i,1]
    }
    for(i in (num_min+1):(num_pred+num_min)){
      if(i > 365){
        reg_new <- data.frame(cbind(1, matrix(fc[1,(i-num_ar):(i-1)], nrow=1),
                                    fc[i-7], fc[i-365],
                                    matrix(df_new[i-num_min,], nrow=1)))
      }else{
        reg_new <- data.frame(cbind(1, matrix(fc[1,(i-num_ar):(i-1)], nrow=1),
                                    fc[i-7], df[length(df[,1])-365+i-num_min-1,1],
                                    matrix(df_new[i-num_min,], nrow=1)))
      }
      fc[i] <- predict(ann, reg_new, type=c("raw"))
    }
    return(fc[1,-(1:num_min)])
  }
  
  # ff_ann
  # Forecasts time series using caret's avNNet, which averages for several
  # 2-layer feed-forward nets with random initializations. Uses STL decomposition
  # to remove patterns on the weekly and annual scales and can also rank reduce the
  # data using SVD if desired. Autoregression values are the main input with
  # dummy vectors for special days serving as additional regressors.
  ff_ann <- function(num_ar, num_hidden){
    rev_ts <- ts(rev, frequency=7)
    rev_stl <- stl(rev_ts, s.window="period")
    stl_weekly <- rev_stl$time.series[,1]
    rev_ns <- rev_stl$time.series[,2] + rev_stl$time.series[,3]
    if(num_ar > 7){
      num_min <- num_ar
      t <- matrix(0,start-1-num_ar,num_ar)
      for(i in 1:num_ar){
        t[,i] <- rev_ns[(num_ar-i+1):(start-i-1)]
      }
    }else{
      num_min <- 7
      if(num_ar > 0){
        t <- matrix(0,start-1-num_min,num_ar)
        for(i in 1:num_ar){
          t[,i] <- rev_ns[(num_min-i+1):(start-i-1)]
        }
      }else{
        t <- rep(0,length(1:(start-1)))
      }
    }
    y <- rev_ns[(num_min+1):(start-1)]
    weekly <- rev_ns[(num_min-7+1):(start-7-1)]
    annual <- c(rev_ns[(1+num_min):(365+num_min)], rev_ns[(1+num_min):(start-1-365)])
    df <- data.frame(matrix(c(y, t, weekly, annual,
                              special_days[(num_min+1):(start-1),]), nrow=start-1-num_min))
    wts_init <- rowSums(special_days[(num_min+1):(start-1),])*100+1
    
    # optional rank reduction; doesn't always help
    # num_comp <- length(df[1,])
    # df <- rank_reduce(df, num_comp)
    
    ann <- avNNet(X1 ~ ., data=df, repeats=10, weights=wts_init,
                  size=num_hidden, linout=TRUE, trace=FALSE, maxit=100, decay=1e-3)
    df_new <- matrix(c(special_days[start:end,]), nrow=num_pred)
    pred <- ann_iterative(ann, num_ar, num_min, df, df_new, num_pred)
    
    # restore weekly seasonality
    stl_weekly_rep <- rep(stl_weekly[(start%%7+7):((start%%7)+6+7)],ceiling(num_pred/7))
    pred <- pred + stl_weekly_rep[1:num_pred]
    
    # plot result and compute the average percent error
    plot_pred(pred, rev_actual, start, end, "Neural Net - ",
              "AR:", num_ar, "H:", num_hidden, "", "")
    error <- compute_error(pred, rev_actual, start, end)
    print(paste("Total MSE:", error, sep=" "))

    return(error)
  }
  
  ic <- 0 # index for auto-regression order
  jc <- 0 # index for number of hidden layer nodes
  
  for(i in ar_vals){
    ic <- ic+1
    for(j in hl_vals){
      jc <- jc+1
      print(paste("ANN with AR:", i, "Hidden:", j, sep=" "))
      ann_error[ic, jc, kc] <- ff_ann(i,j)
      cat("\n")
    }
    jc <- 0
  }
}