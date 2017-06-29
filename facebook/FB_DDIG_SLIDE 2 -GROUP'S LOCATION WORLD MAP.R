
install.packages("ggplot2")
install.packages("maptools")
install.packages("maps")
library("ggmap")
library("maptools")
library("maps")


visited <- c("SFO", "LAX", "North Carolina, USA", "Vilnius", "Riyadh", "Amsterdam","Singapore", "India", "Toronto","Dallas","Dubai")
ll.visited <- geocode(visited)
visit.x <- ll.visited$lon
visit.y <- ll.visited$lat
#> dput(visit.x)
#c(-122.389979, 80.249583, -0.1198244, 144.96328, 28.06084)
#> dput(visit.y)
#c(37.615223, 13.060422, 51.5112139, -37.814107, -26.1319199)














#Using GGPLOT, plot the Base World Map
mp <- NULL
mapWorld <- borders("world", colour="gray50", fill="green") # create a layer of borders
mp <- ggplot() +   mapWorld

#Now Layer the cities on top
mp <- mp+ geom_point(aes(x=visit.x, y=visit.y) ,color="blue", size=3) 
mp

