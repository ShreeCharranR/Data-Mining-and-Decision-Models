#install.packages("randtests")

library("randtests")

x <- read.csv("C:\\Users\\Lenovo\\Desktop\\it.csv",header = FALSE)

test <- runs.test(x$V1)
print("The pvalue of the runs test is for inter arrival time ") ; print(test$p.value)
print(test$alternative)

test2 <- bartels.rank.test(x$V1)
print("The pvalue of the runs von neumann test is for inter arrival time ") ; print(test2$p.value)
print(test2$alternative)