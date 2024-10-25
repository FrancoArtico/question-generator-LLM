# Generación de preguntas de lectura comprensiva en español utilizando LLM

**Resumen:** El enfoque de este proyecto es realizar un generador de preguntas de lectura comprensiva de tipo "de respuesta corta" [1] y verdaderos o falsos, basado en el modelo extenso de lenguaje `"Llama-3.1-8B-Instruct"`, el cual trabajará sobre artículos de biografias presentes en Wikipedia. Su objetivo sería reducir la carga horaria de los profesores y servir de ayuda para el aprendizaje de nuevos textos.

**Autores:** *Franco Artico* y *Nain Cadro*

## Objetivos preliminares y revisión

Los objetivos preliminares planteados en el informe preliminar fueron:
1. Detectar limitaciones de LLaMa2 para la generación de preguntas en español.
2. Realizar técnica de few shot prompting y evaluar los resultados obtenidos.
3. Efectuar fine tuning sobre LLaMa2 y buscar mejores resultados respecto al punto anterior.
4. Determinar las mejoras obtenidas en cada enfoque y realizar conclusiones acerca de las mismas.

En general, los objetivos planteados inicialmente se mantuvieron. Algunos objetivos surgieron mientras trabajabamos en el proyecto. Uno de ellos es hacer una investigación exhaustiva de los modelos de lenguaje que soporten la generación de texto en español, dado a que nos vimos obligados a cambiar la familia de versiones del modelo debido a la falta de soporte y benchmarks para el español del modelo elegido previamente a `Llama 3.1`.
Otro objetivo que surgió es definir un tipo de texto específico sobre el cual trabajar las preguntas a generar, esto lo hicimos debido a la inmensa diversidad de tipos de artículos que hay en la Wikipedia. Pensamos que sería mejor abordar un tipo de artículo para luego poder trabajar sobre otros y así obtener mejores resultados.
Por otro lado, decidimos definir lo más preciso posible el tipo de preguntas que deseabamos que el modelo nos devuelva, junto con los críterios que ibamos a usar para definir cuándo una pregunta iba a ser considerada "bien generada" de acuerdo a los objetivos del proyecto.

## Avances en la metodologia

Existen diferentes tecnicas para lograr que un modelo dado realice de forma correcta una tarea especifica.
En este proyecyo se utilizan varias tecnicas diferentes, las cuales se mencionan a continuacion: Few-shot learning y zero-shot learning, fine-tunning, cuantizacion.

#### Zero-shot learning
*Zero-shot learning* (sin ejemplos) es una técnica que consiste en interactuar como el modelo sin incluirle ejemplos ni demostraciones junto con la instrucción de la tarea. Es decir, directamente solicitar la intrucción a realizar sin ejemplos de cómo queremos los resultados [4].

Dado a que inicialmente comenzamos trabajando sobre la versión `"meta-llama/Llama-3.1-8B"` (versión base), la cual no tiene formato para la entrada. Como otros modelos base, pueden ser usados solamente para continuar la seceuencia que se les proporciona [5]. 
En este proyecto utilizamos esta técnica al principio, para lograr que el modelo produzca preguntas. A continuación se muestra una de las entradas que usamos para probar el modelo:

```python
prompt = article[0] + article[1] + "\n\n En base al texto anterior, responder las siguientes 5 preguntas:"
```

dónde `article[0]` y `article[1]` es el primer y segundo párrafo de un artículo de la Wikipedia sobre Gueorgui Vasílievich Chicherin [6].
El modelo respondió a la entrada de la siguiente manera:
```
En base al texto anterior, responder las siguientes 5 preguntas: 
¿Qué año falleció Gueorgui Vasílievich Chicherin?
¿En qué país estaba cuando estalló la Primera Guerra Mundial?
¿Qué posición adoptó Gueorgui Vasílievich Chicherin al inicio de la Primera Guerra Mundial?
¿Cuál fue el cargo que ocupó Gueorgui Vasílievich Chicherin en el gobierno soviético?
¿Qué país relevó a Gueorgui Vasílievich Chicherin de su cargo en el gobierno soviético? 
```
Esta técnica nos permitió testear al modelo de forma rápida y descubrir que esta versión de la familia LLaMa generaba preguntas que se acercaban al formato de las preguntas típicas de un cuestionario de evaluacion.

#### Few-shot learning (aprendizaje con pocos ejemplos).

Few-shot learining es una técnica de aprendizaje automático en la cual se le pasa al modelo junto con la instrucción algunos ejemplos, generalemente pocos, que servirán como guía para aprender el patrón de los resultados que deseamos [3].

Como mencionamos el subtítulo anterior, comenzamos trabajando con la versión del modelo base pero nos dimos cuenta que la versión *finetuneada* para recibir instrucciones `"meta-llama/Llama-3.1-8B-Instruct"` le podíamos sacar más provecho para hacer prompting debido a que estaba adaptado para recibir instrucciones y usar roles en los mensajes que se le pasan al modelo para especificar contextos y guias. Entre esos roles, destacamos *system* que sirve para establecer el contexto de una conversación o información necesaria que ayude al modelo a responder efectivamente y *user* para denotar la entrada del usuario [5].

Es así como, con instrucciones más precisas y además el uso de roles, armamos una nueva entrada enfocada en esta técnica proporcionando algunos ejemplos de preguntas realizadas por nosotros.

```python
prompt = [
    {"role":"system", "content": '''Eres un generador de preguntas de lectura comprensiva y de tipo verdadero falso, sin generar las respuestas. Algunas de las preguntas que puedes hacer es:  Preguntas de tipo respuestas cortas:
     1. ¿Cuándo y dónde nació?
     2. ¿En que fecha murió y cuál fue la causa de su muerte?
     3. ¿Cuáles eran sus ocupaciones?
     4. ¿Dónde estudió?
     Verdadero o Falso:
     5. Fue un criptogŕafo reconocido, ¿Verdadero o Falso?.
     6. No contribuyó a la Teoría de la Relatividad de Einstein, ¿Verdadero o Falso?'''},
    {"role":"user", "content": "Hacer 5 preguntas de lectura comprensiva de tipo de respuesta corta y 5 de tipo verdadero o falso, sobre el siguiente texto:\n" + godel}
    ]
```
En este caso, la salida del modelo fue la siguiente:

```
Preguntas de lectura comprensiva de tipo de respuesta corta:

1. ¿En qué ciudad nació Kurt Friedrich Gödel?
2. ¿Cuál fue el tema elegido por Gödel para su disertación doctoral?
3. ¿Qué título recibió Gödel en 1933 en la Universidad de Viena?
4. ¿Qué título recibió Gödel en 1946 en el Instituto de Estudios Avanzados?
5. ¿Qué título recibió Gödel en 1976 en el Instituto de Estudios Avanzados?

Preguntas de lectura comprensiva de tipo verdadero o falso:

1. Fue un lógico, matemático y filósofo austríaco, ¿Verdadero o Falso?
2. No contribuyó a la Teoría de la Relatividad de Einstein, ¿Verdadero o Falso?
3. Fue un criptogŕafo reconocido, ¿Verdadero o Falso?
4. No estudió en la Universidad de Viena, ¿Verdadero o Falso?
5. No recibió el Premio Albert Einstein en 1951, ¿Verdadero o Falso?
```

Se intentó mostrarle un ejemplo de la tarea en el rol de usuario, como resultado el modelo interpretaba las preguntas como si tuviese que responderlas, a veces generando preguntas nuevas y a veces no.
Cambiamos este enfoque poniendo los ejemplos en el role system y los resultados mejoraron notablemente debido a que obtuvimos preguntas sin mostrarse sus respuestas y enfocado en ejemplos proporcionados en la entrada.
Un problema que se observo es que al mencionar en la instruccion que el tipo de la pregunta debia ser de "respuesta corta", dejaba de interpretar la parte de la instruccion que pedia que no muestre las respuestas. Luego nos dimos cuenta que estabamos ingresando las instrucciones en el role user y debia colocarse en el rol system, ya que dicho rol es el que guia al modelo en como realizar la tarea.

#### Cuantización
Es una técnica usada en machine learning para reducir el uso de los recursos computacionales y de memoria de los modelos grandes de lenguaje, para hacerlos más eficientes [2].
Se decidió aplicar esta técnica a los modelos probados, debido a que incluso con versiones de pocos parámetros, la ejecución tardaba demasiado o incluso excedía los recursos disponibles en nuestro entorno de ejecución. A través de la cuantización de 4 bits pudimos ejecutar los modelos más pequeños de la familia LLaMa como ser las versiones `Llama2-7B` y `Llama3-8B` y también Llama3-8B-Intruct, donde este ultimo tiene más párametros.

## Problemas encontrados

A continuación nombraremos algunos de los problemas encontrados durante el desarrollo del proyecto y cómo se abordaron.

**Recursos y entorno de ejecución:** utilizamos Google Colab debido a su fácil acceso y adecuación para probar versiones de pocos parámetros de modelos extensos de lenguaje. Sin embargo, los recursos ofrecidos por este entorno comienzan a limitarnos a medido que el proyecto va avanzando en complejidad. Esto lo abordaremos, usando un nuevo entorno de ejecución más potente (CCAD) brindado por la UNC.

**Elección del tipo de preguntas a generar**: empezamos probando el modelo para la tarea de generar preguntas en base a un artículo, pero no definimos previamente el tipo de preguntas que esperabamos, esto generó incertidumbre en el momento de definir cuándo estabamos mejorando la calidad de las preguntas. Para este problema lo que hicimos fue definir una serie de pautas enfocados en las características de las preguntas:
* *Tipo de preguntas:* seleccionamos preguntas de tipo "respuestas cortas" y de tipo "verdadero o falso". Las respuestas a las preguntas de tipo "respuesta corta" deben ser de una palabra u oración, y las respuestas a ambos tipos deben estar presentes en el artículo. La motivación en la elección de cada tipo fue:
    * Respuesta corta: mayor retroalimentación y mayor sencillez en la corrección que otro tipo de preguntas (tipo abiertas por ejemplo). 
    * VoF: facilidad de corrección.
* *No repetición:* buscamos diversidad en preguntas, por lo tanto dos preguntas de la misma lista no deben ser idénticas y tampoco demasiado similares. A continuación dejamos un ejemplo de preguntas similares que queremos evitar
  ```
   Preguntas 1: ¿Que hizo en el Siglo 19?
   Preguntas 2: ¿Que hizo en el Siglo 20?
  ```
* *Naturalidad y fluidez:* las preguntas generadas no deben mostrar un estructura mecanizada y evitando repeticiones innecesarias, así como si hubiesen sido hechas por una persona.
* *Variedad:* priorizaremos la relevancia pero tendremos en cuenta que sean abarcativas a los temas tratados en el articulo.
* *Grado de dificultad:* buscamos que se generen preguntas para niveles de comprensión desde básico a intermedio.
* *Orientadas al refuerzo de la comprensión *: Las preguntas buscan evaluar a quién las responda en su habilidad para identificar ideas principales y detalles específicos (por ejemplo: fechas).

**Elección del modelo:**

Definir que versión del modelo a usar nos llevó demasiado tiempo debido a que tuvimos que investigar y probar las diferentes posibilidades. El modelo extenso de lenguaje que decidimos utilizar es LLaMa en su versión `meta-llama/Llama-3.1-8B-Instruct`. 
Para tomar esta decisión, lo primero que consideramos fue hacer una evaluación manual sobre las versiones `meta-llama/Llama-2-13b-hf` y `meta-llama/Llama-3.1-8B` acerca de qué tan bien generaban texto en español.

Mediante esta evaluación determinamos que `meta-llama/Llama-3.1-8B` generaba texto en español con coherencía y fluidez, contrario a `meta-llama/Llama-2-13b-hf` que inclusive repetía oraciones generadas, en lugar de generar nuevas, lo que lo volvía poco natural. Un ejemplo del resultado de las pruebas realizadas se muestra a continuación, donde se le pidió a ambos modelos que generen texto a partir de la entrada`"El perro estaba "` con los mismos parámetros para los dos: 

![Compartiva](https://i.imgur.com/UsBXOZ6.png)

Al entrar en detalle sobre los modelos, observamos vía su página web que a diferencia de su versión 3.1, `Llama-2` no había sido entrenado con el objetivo de la generación de texto en múltiples idiomas, lo que fue determinante para dejar de considerarlo como una posibilidad.

Como ya hemos mencionado a lo largo del artículo, la elección del modelo orientado a instrucciones `"meta-llama/Llama-3.1-8B-Instruct"` fue clave para el *prompting* debido a que puede interpretar instrucciones y usar roles en su entrada para adaptar su salida [7].

**Elección tipo de texto:** dado a que decidimos trabajar con artículos de Wikipedia y la misma contiene una extensa cantidad de tipos de artículos se decidió restringir el dominio a biografías. Incluso de esta manera, tenemos el problema de que dos biografías pueden ser muy distintas entre sí respecto a los temas tratados en ellas. 

Pensamos trabajar para la próxima entrega con biografías más variadas (pej. incluyendo personajes históricos de diversas ramas) y manejar los posibles problemas futuros que se presenten.

**Detectar limitaciones:** si bien el modelo con la versión orientada a instrucciones generaba preguntas bastante bien, notamos que bajo la instruccion de "generar preguntas" sin especificar un tipo de las mismas, el modelo solo generaba preguntas de tipo abiertas, lo cual alcanzaba nuestro objetivo parcialmente, ya que no geraba preguntas de tipo verdadero o falso  por defecto. Cabe destacar tambien que las preguntas , en general, no eran de respuesta corta si no de respuestas "abiertas".
    * Intentamos adaptar la complejidad de las preguntas para que no sean ni muy largas de responder, ni muy fáciles.


**El modelo devolvía respuestas no solicitadas:** al presentarse tantas veces la palabra "respuesta" en el prompt el modelo mal interpretaba y generaba respuestas junto con las preguntas cuándo de manera explícita le pedimos que no lo haga. Esto lo abordamos modularizando el prompt en varios mensajes de sistema y vez de hacer un prompt más largo, hicimos uno más corto y específico.

**Preguntas demasiado estructuradas**: buscamos que el modelo genere repuestas naturales y no tán robóticas. Esto pensamos solucionarlo dando más ejemplos proporcionados por nosotros.


**Implementacion de rol system y el rol usuario:** hay información que deseabamos introducirle al modelo como ejemplos de tipo de preguntas o el texto mismo, las cuales no sabíamos en cuál rol suministrarselas. Para solucionarlo, hicimos diversas pruebas pasando la información en ambos roles y nos quedamos con el promting que mejor resultados nos otorgaba.


**Enfoque del few-shot:** tenemos dos enfoques posibles pasarle varios ejemplos junto con los artículos o solo ejemplos de preguntas generales sobre biografías que se podrían realizar. Para esto, pensamos hacer una evaluación sobre ambos enfoques.

**Fine tuning:** aún no hemos investigado cómo hacerlo, por lo cual pensamos investigar más sobre como hacerlo en este modelo a través de trabajos previos o mismo recursos disponibles en la web.

##  Evaluacion del progreso
Hasta ahora se realizó una evaluación anedótica manual mediante comparaciones entre los diferentes resultados obtenidos cambiando las entradas y las técnicas utilizadas. Se planea para un futuro investigar alternativas de evaluacion.


## Planificación actualizada

* Fine tuning (Semana del 27/10)
    * Preparar el conjunto de datos para usar la técnica.
        * Determinar la forma que tendrá nuestro conjunto de datos.
        * Seleccionar los artículos sobre los cuales haremos las preguntas.
        * Hacer las preguntas.
        * Limpiar el conjunto de datos para su posterior utilización.
    * Realizar fine tuning del modelo con el conjunto de datos.
    * Evaluar las preguntas generadas y comparar con los resultados previos logrados con el resto de las técnicas.
    * Documentar el proceso, observaciones y resultados. 
    * Elaborar una conclusión donde se comparen los resultados y el costo de aplicar cada una de las técnicas.
    * Hacer correcciones finales sobre las netbooks.

****************************************

### Nuevas referencias

1. https://urjconline.atavist.com/2023/06/29/pregunta-tipo-respuesta-corta/.
2. https://www.llama.com/docs/how-to-guides/quantization/.
3. https://www.llama.com/docs/how-to-guides/prompting.
4. https://www.promptingguide.ai/techniques/zeroshot.
5. https://huggingface.co/blog/llama31#how-to-prompt-llama-31.
6. https://es.wikipedia.org/wiki/Gueorgui_Chicherin.
7. https://huggingface.co/blog/llama31#using-hugging-face-transformers.