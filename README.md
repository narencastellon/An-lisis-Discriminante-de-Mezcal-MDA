# Analisis-Discriminante-de-Mezcal-MDA

## **5. Análisis discriminante de mezcla - MDA.**

El clasificador LDA supone que cada clase proviene de una única distribución normal (o gaussiana). Esto es demasiado restrictivo.

Para MDA, hay clases, y se supone que cada clase es una mezcla gaussiana de subclases, donde cada punto de datos tiene una probabilidad de pertenecer a cada clase. Aún se asume la igualdad de la matriz de covarianza, entre clases.

## **Paso 1. Carga de paquetes R requeridos.**
Carga de paquetes R requeridos

`tidyverse` para una fácil visualización y manipulación de datos.
`caret` para un flujo de trabajo de aprendizaje automático (Machine Learning) sencillo.

```{r message=FALSE}
library(tidyverse)
library(caret)
library(klaR)
theme_set(theme_classic())
```


## **Paso 2. Preparando los datos.**

Usaremos el conjunto iris de datos, para predecir especies de iris basadas en las variables predictoras Sepal.Length, Sepal.Width, Petal.Length, Petal.Width.

El análisis discriminante puede verse afectado por la escala / unidad en la que se miden las variables predictoras. Generalmente se recomienda estandarizar / normalizar el predictor continuo antes del análisis.

**2.1. Divida los datos en entrenamiento y conjunto de prueba:**

```{r}
# Cargamos la data
data("iris")
# Dividimos la data para entrenamiento en un (80%) y para la prueba en un (20%)
set.seed(123)
training.samples <- iris$Species %>%
createDataPartition(p = 0.8, list = FALSE)
train.data <- iris[training.samples, ]
test.data <- iris[-training.samples, ]
```

**2. Normaliza los datos. Las variables categóricas se ignoran automáticamente.**

```{r}
# Estimar parámetros de preprocesamiento
preproc.param <- train.data %>% 
preProcess(method = c("center", "scale"))
# Transformar los datos usando los parámetros estimados
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)
```

# **Paso 3. Creación del Modelo MDA**

```{r warning=FALSE,message=FALSE}
library(class)
library(mda)
# Creando el modelo
modelmda <- mda(Species~., data = train.transformed)
modelmda
```

## **Paso 4. Gráficos de partición MDA**

```{r}
library(klaR)
partimat(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=train.transformed, method="mda")
```
## **Paso 5: use el modelo para hacer predicciones MDA**

Una vez que hemos ajustado el modelo utilizando nuestros datos de entrenamiento, podemos usarlo para hacer predicciones sobre nuestros datos de prueba:

```{r}
# Haciendo predicciones
predicted.classes <- modelmda %>% predict(test.transformed)
```

## **Paso 6: evaluar el modelo MDA**

Podemos usar el siguiente código para ver para qué porcentaje de observaciones el modelo QDA predijo correctamente la Specie:

```{r warning=FALSE,message=FALSE}
# Precisión del Modelo
mean(predicted.classes == test.transformed$Species)
```

MDA podría superar a LDA y QDA en algunas situaciones, como se ilustra a continuación. En este ejemplo de datos, tenemos 3 grupos principales de individuos, cada uno de los cuales no tiene 3 subgrupos adyacentes. Las líneas azul continuas en el gráfico representan los límites de decisión de LDA, QDA y MDA. Se puede observar que el clasificador MDA ha identificado correctamente las subclases en comparación con LDA y QDA, que no fueron nada buenas en la modelización de estos datos.

```{r warning=FALSE, message=FALSE}
#install.packages("mda")
#install.packages("mvtnorm")
library(mda)
library(mvtnorm)
library(MASS)

#Muestra de datos aleatoriamente
set.seed (42)
n <- 500
x11 <- rmvnorm (n = n, mean = c (-4, -4))
x12 <- rmvnorm (n = n, mean = c (0, 4)) 
x13 <- rmvnorm (n = n, mean = c (4, -4))

x21 <- rmvnorm (n = n, mean = c (-4, 4)) 
x22 <- rmvnorm (n = n, mean = c (4, 4)) 
x23 <- rmvnorm (n = n, mean = c ( 0, 0))

x31 <- rmvnorm (n = n, mean = c (-4, 0))
x32 <- rmvnorm (n = n, mean = c (0, -4)) 
x33 <- rmvnorm (n = n, mean = c (4, 0))

x <- rbind (x11, x12, x13, x21, x22, x23, x31, x32, x33) 
train_data <- data.frame (x, y = gl (3, 3 * n))

#Creación de Modelos
lda_out <- lda (y ~., data = train_data) 
qda_out <- qda (y ~., data = train_data)
mda_out <- mda (y ~., data = train_data)

#Genera datos de prueba que se utilizarán para generar los límites de decisión mediante
# contours
contour_data <- expand.grid (X1 = seq (-8, 8, length = 300),
                             X2 = seq (-8, 8, length = 300))

#Clasifica los datos de prueba
lda_predict <- data.frame (contour_data, y = as.numeric (predict (lda_out, contour_data) $ class))
qda_predict <- data.frame (contour_data, y = as.numeric (predict (qda_out, contour_data) $ class))
mda_predict <- data.frame (contour_data, y = as.numeric (predict (mda_out, contour_data)))

#Genera gráficos
library(ggplot2)
library(ggpubr)# Nos permite unir varios grafico usando la función ggarrange()

p1<-ggplot (train_data, aes (x = X1, y = X2, color = y)) +
  geom_point ()+ stat_contour (aes (x = X1, y = X2, z = y), 
                data = lda_predict) + ggtitle ("Límites de decisión\n de LDA")
 
p2<-ggplot (train_data, aes (x = X1, y = X2, color = y)) +
  geom_point ()+
  stat_contour (aes (x = X1, y = X2, z = y), data = qda_predict) +
  ggtitle ("Límites de decisión\n de QDA")
 
p3<-ggplot (train_data, aes (x = X1, y = X2, color = y)) +
  geom_point ()+
     stat_contour (aes (x = X1, y = X2, z = y), data = mda_predict) + 
  ggtitle ("Límites de decisión de la MDA") 

ggarrange(p1, p2,p3)

```
