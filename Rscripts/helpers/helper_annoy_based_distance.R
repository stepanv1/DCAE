# Annoy based k-nearest neighbours learning
cat("Performing fast PCA.\n")

data("iris")

library(RcppAnnoy)

set.seed(123)                           # be reproducible

f <- 40 #dimensions
a <- new(AnnoyEuclidean, f)
n <- 50 #number of objects             # not specified

for (i in seq(n)) {
  v <- rnorm(f)
  a$addItem(i-1, v)
}

a$build(50)                           	# 50 trees
a$save("/tmp/test.tree")


b <- new(AnnoyEuclidean, f)           	# new object, could be in another process, with dimensionalitu of vector space f
b$load("/tmp/test.tree")		# super fast, will just mmap the file

print(b$getNNsByItem(0, 40))




X = t(Reduce(cbind, (lapply(0:(n-1), function(x) a$getItemsVector(x)))))
k=10

cat("Performing k-nearest neighbour search.\n")

annoyKnn <- function(X, k, ntree=200){
  a_annoy = new(AnnoyEuclidean,dim(X)[2])
  n_annoy = dim(X)[1]
  for (i_annoy in seq(n_annoy)) {
    v_annoy = as.vector(X[i_annoy,])
    a_annoy$addItem(i_annoy-1,v_annoy)
  }
  a_annoy$build(ntree)
  val = array(0,c(dim(X)[1],k*2))
  ind = array(0,c(dim(X)[1],k*2))
  for(j_annoy in 1:dim(val)[1]) {
    ind[j_annoy,] = a_annoy$getNNsByItem(j_annoy-1,k*2)+1
    for(val_annoy in 1:dim(val)[2]) {
      val[j_annoy,val_annoy] = a_annoy$getDistance(j_annoy-1,ind[j_annoy,val_annoy]-1)
    }
  }
  return(list('ind' = ind, 'dist' = val))
}