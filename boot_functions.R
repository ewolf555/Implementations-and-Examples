# Functions to obtain small sample adjusted confidence intervals for VAR Models 
# as given by Kilian (1995)


resamp <- function (e) 
{
  n <- nrow(e)
  random <- runif(n)
  index <- as.integer(1 + n * random)
  es <- e[index, , drop = FALSE]
  return(es)
}

VAR.ys <- function(x, b, p, e, type){
  n <- nrow(x)
  k <- nrow(b)
  b0 <- b[, 1:(k * p), drop = FALSE]
  if (type == "const") 
    b1 <- as.matrix(b[, (k * p) + 1], nrow = k)
  if (type == "const+trend") {
    b1 <- as.matrix(b[, (k * p) + 1], nrow = k)
    b2 <- as.matrix(b[, (k * p) + 2], nrow = k)
  }
  y <- x[1:p, , drop = FALSE]
  for (i in (p + 1):n) {
    index <- 1:k
    ytem <- y[nrow(y):1, , drop = FALSE]
    d1 <- 0
    for (j in 1:p) {
      d <- b0[, index] %*% t(ytem[j, , drop = FALSE])
      d1 <- d1 + d
      index <- index + k
    }
    d1 <- d1 + as.matrix(e[i - p, ], nrow = k)
    if (type == "const") 
      d1 <- d1 + b1
    if (type == "const+trend") 
      d1 <- d1 + b1 + b2 * i
    colnames(d1) <- NULL
    y <- rbind(y, t(d1))
  }
  return(y)
}

VAR.names <- function(x, p, type = "const"){
  tem1 <- colnames(x)
  varnames <- character()
  for (i in 1:p) {
    tem2 <- paste(tem1, rep(-i, ncol(x)), sep = "(")
    tem3 <- paste(tem2, ")", sep = "")
    varnames <- c(varnames, tem3)
  }
  if (type == "const") 
    varnames <- c(varnames, "const")
  if (type == "const+trend") 
    varnames <- c(varnames, "const", "trend")
  return(varnames)
}

VAR.resid <- function(x, b, z, p){
  n <- nrow(x)
  k <- ncol(x)
  y <- t(x[(p + 1):n, ])
  e <- t(y - b %*% z)
  tem <- colMeans(e)
  es <- matrix(NA, nrow = nrow(e), ncol = ncol(e))
  for (i in 1:k) {
    es[, i] <- matrix(e[, i] - mean(e[, i]))
  }
  return(es)
}

varSim <- function (x, p, type = "const"){
  
  n <- nrow(x)
  k <- ncol(x)
  var1 <- VAR.etp::VAR.est(x, p, type)
  b <- var1$coef
  e <- sqrt((n - p)/((n - p) - ncol(b))) * var1$resid
  mat <- matrix(0, nrow = k, ncol = ncol(b))
  es <- resamp(e)
  xs <- VAR.ys(x, b, p, es, type)
  bs <- VAR.etp::VAR.est(xs, p, type)$coef
  colnames(bs) <- VAR.names(x, p, type)
  es <- VAR.resid(x, bs, var1$zmat, p)
  colnames(es) <- rownames(b)
  sigu <- t(es) %*% es/((n - p) - ncol(b))
  return(list(coef = bs, resid = es, sigu = sigu))
} 
