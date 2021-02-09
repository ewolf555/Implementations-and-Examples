# SVAR Analysis baes in the Paper by Valentina Colombo (2013)

library(readr)
library(readxl)
library(vars)
library(VAR.etp)
library(dplyr)


setwd("...")

# Import functions
source("boot_functions.R")

# Import Data
cpiEU <- read_csv("Daten/CPI_Euro.csv")
ipiEU <- read_csv("Daten/IPI_EU.csv")
cpiUS <- read_excel("Daten/CPIAUCSL.xls")
iUS <- read_excel("Daten/FEDFUNDS.xls")
ipiUS <- read_excel("Daten/INDPRO.xls")
iEU <- read_excel("Daten/IR3TIB01EZM156N.xls")
newsUS <- read_excel("Daten/US_Policy_Uncertainty_Data_News.xlsx")
newsEU <- read_excel("Daten/Europe_Policy_Uncertainty_Data_News.xlsx")


# Adjust Data lengths and transform to loglevels
Date <- cpiEU$DATE[13:236]
cpiEU <- log(cpiEU$CP0000EZ19M086NEST[13:236])
ipiEU <- log(ipiEU$IPI_EU[31:254])
cpiUS <- log(cpiUS$CPIAUCSL[601:824])
ipiUS <- log(ipiUS$INDPRO[937:1160])
iUS <- iUS$FEDFUNDS[511:734]
iEU <- iEU$IR3TIB01EZM156N[37:260]
newsUS <- newsUS$News_Based_Policy_Uncert_Index[145:368]
newsEU <- newsEU$European_News_Index


varDataFull <- data.frame(Date = Date,
                          cpiUS = cpiUS,
                          ipiUS = ipiUS,
                          iUS = iUS,
                          newsUS = newsUS,
                          cpiEU = cpiEU,
                          ipiEU = ipiEU,
                          iEU = iEU,
                          newsEU = newsEU)

varData <- varDataFull[25:224, ] %>% select(2:9)

varNames <- colnames(varData)
plotNames <- c("Consumer Price Index", "Industrial Production Index", "Interest Rate", "News Index")
plotNamesShort <- c("CPI", "IPI", "Interest Rate", "News Ind.")
pos <- c("left", "topleft", "bottomleft", "topleft")
xTicks <- seq(from = 1, to =length(varData$cpiUS), 60)
tickLabels <- c("1997", "2002", "2007", "2012")

#write.csv(varData, file = "~/Uni/Multivariate Timeseries/Daten/varData.csv", row.names = FALSE)

# Variable Plots and descriptive Stats
for (i in 1:4){
  seriesEU <- varData[[i+4]]
  seriesUS <- varData[[i]]
  yMax <- max(c(max(seriesEU), max(seriesUS)))
  yMin <- min(c(min(seriesEU), min(seriesUS)))         
  plot(seriesEU, type = "l",  ylab = plotNamesShort[i], xlab = "Year", 
       main = paste(plotNames[i], sep = " "),
       ylim=c(yMin, yMax), lwd = 2, xaxt = "n")
  lines(seriesUS, type = "l", col = "blue", lwd = 2)
  grid()
  axis(side=1, at = xTicks, labels = tickLabels)
  legend(pos[i], legend = c("EU", "US"), col=c("black", "blue"), 
         lty = 1, lwd = 2)

}

summary(varData)

############################### Estimation #####################################

# Var selection
infCrit <- VARselect(varData, lag.max = 15, type = "both")
print(infCrit$selection)
bestModel <- infCrit$selection["HQ(n)"]
print(bestModel)


# Var(3) estimation
var1 <- VAR(varData, p = bestModel, type = "both")
summary(var1)


print(roots(var1) < 1)



################################################################################

irfDist <- list(cpiUS = matrix(ncol = 2000, nrow = 26),
                ipiUS = matrix(ncol = 2000, nrow = 26),
                iUS = matrix(ncol = 2000, nrow = 26),
                newsUS = matrix(ncol = 2000, nrow = 26),
                cpiEU = matrix(ncol = 2000, nrow = 26),
                ipiEU = matrix(ncol = 2000, nrow = 26),
                iEU = matrix(ncol = 2000, nrow = 26),
                newsEU = matrix(ncol = 2000, nrow = 26))

irfNames <- names(irfDist)

bias <- VAR.Boot(as.matrix(varData), p = bestModel, nb = 2000, type = "const+trend")

# Impulse response bootstrap
for (i in 1:2000){
  
  boot <- varSim(varData, p = bestModel, type = "const+trend")
  
  for (j in ncol(boot$coef)){
    boot$coef[ ,j] <- boot$coef[ ,j] - bias$Bias[,j]
  }
  
  modVar <- var1
  
  for (j in 1:nrow(bias$coef)){
    coefNames <- names(var1$varresult[[j]]$coefficients)
    modVar$varresult[[j]]$coefficients <- boot$coef[j, ]
    names(modVar$varresult[[j]]$coefficients) <- coefNames
  }

  irfboot <- irf(modVar, ortho = TRUE, n.ahead = 25, 
                 impulse = "newsUS", boot = FALSE)  
  
  for (j in 1:length(irfNames)){
    irfDist[[j]][ ,i] <- irfboot$irf$newsUS[ ,j]
  }
}

irfVar <- list(cpiUS = matrix(ncol = 3, nrow = 26),
               ipiUS = matrix(ncol = 3, nrow = 26),
               iUS = matrix(ncol = 3, nrow = 26),
               newsUS = matrix(ncol = 3, nrow = 26),
               cpiEU = matrix(ncol = 3, nrow = 26),
               ipiEU = matrix(ncol = 3, nrow = 26),
               iEU = matrix(ncol = 3, nrow = 26),
               newsEU = matrix(ncol = 3, nrow = 26))

for (i in 1:8){
  for (j in 1:26){
    irfVar[[i]][j, 1] <- quantile(irfDist[[i]][j, ], 0.05)
    irfVar[[i]][j, 2] <- median(irfDist[[i]][j, ])
    irfVar[[i]][j, 3] <- quantile(irfDist[[i]][j, ], 0.95)
  }
}

irfNames <- c("CPI (US)", "IPI (US)", "I (US)", "NEWS (US)",
              "CPI (EU)", "IPI (EU)", "I (EU)", "NEWS (EU)")

par(mfrow = c(1,1))
for (i in 1:8){
  plot(irfVar[[i]][,2], type = "l",
       ylim = c(min(irfVar[[i]][,1]), max(irfVar[[i]][,3])), 
       xlab = "Periods", ylab = irfNames[i])
  lines(irfVar[[i]][,1], type = "l", col = "blue")
  lines(irfVar[[i]][,3], type ="l", col = "blue")
  abline(h = 0, col = "red")
  grid()
}



fevdTables <- fevd(var1, n.ahead = 24)

rowInd <- c(1, 6, 12, 18, 24)
colInd <- c(5, 5, 6, 6, 7, 7)
listInd <- c(8, 4, 8, 4, 8, 4)

fevdTable <- matrix(ncol = 6, nrow = 24)

for (i in 1:6){
  fevdTable[i, ] <- fevdTables[[listInd[i]]][rowInd[i] ,colInd[i]]
}


fevdTable[ ,1] <- fevdTables$newsEU[ ,5]
fevdTable[ ,2] <- fevdTables$newsUS[ ,5]
fevdTable[ ,3] <- fevdTables$newsEU[ ,6]
fevdTable[ ,4] <- fevdTables$newsUS[ ,6]
fevdTable[ ,5] <- fevdTables$newsEU[ ,7]
fevdTable[ ,6] <- fevdTables$newsUS[ ,7]

fevdTable <- fevdTable[rowInd, ]

rownames(fevdTable) <- as.character(rowInd)

print(round(fevdTable, digits = 4)*100)



