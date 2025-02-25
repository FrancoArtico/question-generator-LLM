# Generación de preguntas de lectura comprensiva en español utilizando LLM

#### Resumen
Este proyecto presenta una prueba de concepto para un generador de preguntas de lectura comprensiva en español, que se enfocará principalmente en dos tipos de preguntas: de respuesta corta [1] y afirmaciones a evaluar como verdaderas o falsas. Para ello, utilizaremos el modelo de lenguaje `"Llama-3.1-8B-Instruct"` aplicando distintas técnicas y evaluando sus posibles mejorías en el desempeño. A lo largo de este proyecto trabajaremos principalmente sobre artículos de Wikipedia. Consideramos este estudio de interés dado a que la viabilidad de un sistema con estas características podría reducir la carga horaria de profesores y reforzar el aprendizaje de los estudiantes sobre los textos.

**Autores:** *Franco Artico* y *Nain Cadro*.

## Hipótesis, objetivos iniciales y el estado final alcanzado
Las **hipótesis** sobre las cuales se desarrolló este proyecto son:
* El modelo presenta fallas en la generación de preguntas en español, las cuales podrían ser errores sintácticos o preguntas desalineadas con los objetivos del proyecto.
* La reducción del dominio del problema nos permitirá identificar más errores y mejorar la calidad de las preguntas generadas.

Los **objetivos preliminares** planteados fueron:
*  Detectar limitaciones del modelo elegido para la generación de preguntas en español.
* Aplicar *few shot prompting* y evaluar los resultados obtenidos.
* Efectuar *fine tuning* sobre el modelo y buscar mejores resultados respecto al punto anterior.
* Determinar las mejoras obtenidas en cada enfoque y realizar conclusiones acerca de las mismas.

Sin embargo, a lo largo del desarrollo del proyecto fueron surgiendo los siguientes objetivos 

* Realizar una invesigación exhaustiva de los modelos de lenguaje que fueron entrenados para generar texto en español.
* Aplicar *zero-shot prompting* y evaluar los resultados obtenidos.
* Seleccionar el tipo de texto específico sobre el cual el modelo realizará preguntas.
* Definir críterios de evaluación para las preguntas generadas por el modelo de acuerdo con el objetivo del proyecto.
* Elegir un conjunto de datos apropiado para realizar el *fine tuning* del modelo.


## Técnicas relevantes

Existen diferentes técnicas para lograr que un modelo realice de forma correcta una tarea específica. En este proyecto se abordaron las siguientes técnicas: *zero-shot learning*, *few-shot learning*, *system prompt*, *fine tuning* y cuantización.

### Zero-shot learning
*Zero-shot learning* (sin ejemplos) es una técnica que consiste en interactuar con el modelo proporcionando la instrucción a realizar sin incluir ejemplos ni demostraciones de cómo queremos que sea realizada [4].

Esta técnica la utilizamos inicialmente buscando descubrir la clase de preguntas generadas por los modelos base de la familia Llama 3.1, los cuales no poseen un entrenamiento específico en esta tarea.

Un modelo base no sigue instrucciones. Fue entrenado sobre una gran cantidad de datos para adquirir conocimientos básicos acerca del lenguaje, conocimiento general y contextos, por lo que es útil en situaciones donde debe dar respuestas de conocimiento general o conversaciones casuales. Sin embargo, para obtener lo que deseamos en este proyecto debemos solicitarlo de forma diferente, agregando indicaciones extras o guías adicionales [10].

A continuación, presentamos un ejemplo de entrada que usamos para probar el modelo `meta-llama/Llama-3.1-8B`:

```python
prompt = article[0] + article[1] + "\n\n En base al texto anterior, responder las siguientes 5 preguntas:"
```

siendo `article[0]` y `article[1]` el primer y segundo párrafo respectivamente de un artículo de Wikipedia. De esta manera, incentivamos al modelo a generar preguntas respecto al texto que le proporcionamos.

### Few-shot learning

*Few-shot learining* es una técnica que consiste en pasarle al modelo, junto con la instrucción de la tarea, unos pocos ejemplos que sirvan como guía para aprender el patrón de los resultados que deseamos obtener [3].

La idea de utilizar esta técnica en nuestro proyecto surge, principalmente, al darnos cuenta que los modelos naturalmente no generan preguntas de verdadero o falso. Entonces, necesitábamos proporcionarles ejemplos para incentivarlos a generarlas.

Un ejemplo de entrada que le proporcionamos al modelo usando este enfoque es el siguiente:

```python=
Agregar ejemplo, modelo base con técnicas de few shot donde se agregen ejemplos de verdaderos o falsos
```

### System prompts

*System prompts* son un conjunto de instrucciones, guías e información acerca del contexto que se le provee al modelo antes de interactuar con las consultas del usuario [11].

El uso de modelos base para la generación de preguntas presenta varias limitaciones. En primer lugar, no permite especificar de manera directa la cantidad y el tipo de preguntas a generar. Además, requieren estrategias más avanzadas de ingeniería de *prompts* para guiar de forma adecuada la generación de las preguntas, lo que implica un mayor esfuerzo en el diseño de instrucciones óptimas. Por estos motivos, se decidió optar por un modelo instruido como `meta-llama/Meta-Llama-3.1-8B-Instruct`, lo que también tendrá un impacto positivo en el *fine tuning*, ya que nos permite enfocarnos en mejorar el desempeño en la tarea específica, sin la necesidad de enseñarle al modelo a interpetar las instrucciones desde cero.

El cambio de modelo a `"meta-llama/Llama-3.1-8B-Instruct"` nos permitió aplicar esta técnica en nuestro proyecto, debido a que cuenta con ciertos roles, entre los cuales destacamos *system* y *user*. El rol *system* sirve para establecer el contexto de una conversación o proveerle al modelo información necesaria para ayudarle a responder efectivamente, mientras que el rol *user* representa la entrada del usuario [5].

El siguiente fragmento de código muestra cómo definimos los roles `system`  y `user` en nuestras consultas al modelo:

```python
prompt = [
    {
        "role": "system",
        "content": (
            "Eres un generador de preguntas de lectura comprensiva. "
            "Tu tarea es crear preguntas basadas en un texto dado, "
            "sin incluir respuestas ni opciones. Solo debes generar preguntas, "
            "en las cuales deberás incluir preguntas de tipo verdadero o falso."
        ),
    },
    {
        "role": "user",
        "content": (
            "Genera exactamente 10 preguntas, de las cuales 5 deben ser de tipo "
            "verdadero o falso, sin incluir las respuestas ni opciones, sobre el siguiente texto:\n\n"
            f"{article}"
        ),
    },
]
```

### Cuantización
La cuantización es una técnica que nos permite comprimir modelos extensos de lenguaje representando sus pesos y activadores, los cuales están típicamente en números flotantes de 32-bits de precisión, con valores de menor precisión, usualmente 8-bits o 4-bits. La reducción en el número de bits provoca una gran disminución en el tamaño del modelo, por lo que consumen menos memoria, requieren menos espacio de guardado y son más rápidos de aplicarle *fine tuning* [2][12].

![imagen](https://i.imgur.com/3WSwSJF.png)

El modelo `"meta-llama/Llama-3.1-8B-Instruct"` requiere disponer de al menos 16 GB de RAM [13]. Sin embargo, dado que nuestro entorno de ejecución en Google Colab solo ofrecía 15 GB de RAM, se hizo necesario aplicar cuantización para reducir el modelo. Para ello, implementamos la técnica NF4 (*4-bit Normal Float*), la cual cuantifica los pesos a 4 bits, permitiéndonos ejecutar y evaluar distintos modelos (como Llama2-7B, Llama3.1-8B y Llama3.1-8B-Instruct) pese a las limitaciones de recursos del entorno.

Además, un artículo de Hugging Face [14], nos permitió saber antes de entrenar nuestro modelo que sería posible hacerlo en este entorno limitado usando GC (*gradient checkpointing*) y *nested quantization*.

### Fine tuning

*Fine tuning* es el proceso de ajustar los parámetros de un modelo extenso de lenguaje pre-entrenado para una tarea o dominio específico a través de un conjunto de datos. Esto dado que, los grandes modelos poseen un conocimiento amplio pero carecen de especialización en áreas específicas, por lo que esta técnica ataca ese problema haciendo a los modelos más acertados y efectivos para ciertos dominios o aplicaciones [9].

*Supervised Fine Tuning* es un método para mejorar y personalizar modelos pre-entrenados, la cual consiste en volver a entrenar estos modelos en un conjunto de datos etiquetado, compuesto por las instrucciones y las respuestas que se desean. Las tres técnicas más populares para aplicar SFT son *full fine-tuning*, *LoRA* y *QLoRA*.

![imagen](https://i.imgur.com/IG9s8Yk.png)

*QLoRA (Quantization-aware Low-Rank Adaptation)* es una técnica que en lugar de entrenar el modelo entero, congela los pesos e introduce pequeños adaptadores denominados *low-rank matrices* lo que le permite reducir mucho el número de parámetros a entrenar, reduciendo el uso de la memoria y el tiempo de entrenamiento respecto al *full fine-tuning*. Esta técnica es ideal para entornos donde la memoria GPU es limitada y por eso fue la elegida para entrenar nuestro modelo. La siguiente imagen ilustra la reducción de parámetros entrenables que esta técnica realizó en nuestro caso.

![](https://i.imgur.com/vqrHSB7.png)

El conjunto de datos que seleccionamos para esta técnica fue *Spanish Question Answering Corpus* (SQAC) [15], que se compone de preguntas-respuestas extractivas en español. Sin embargo, lo consideramos ideal para la tarea de este proyecto, ya que posee contextos extraídos de artículos de Wikipedia, noticias de WikiNews y la sección española de AnCora corpus que es una mezcla de diferentes fuentes de noticias y literatura. Además, no posee preguntas sin respuestas y tanto las preguntas como las respuestas fueron anotadas por hablantes nativos de español con estudios universitarios [17]. 

Ahora bien, a este conjunto de datos fue necesario realizarle un pre-procesamiento donde solamente nos quedamos con contextos provenientes de la sección "Biografías" de artículos de Wikipedia. Posterior a eso, nos quedamos con las columnas `context` y `question` para, a partir de ellas, armar una sola columna (`text`) representando una conversación entre un usuario que provee un contexto y solicita una cierta cantidad de preguntas, y un asistente que genera las preguntas a partir de ese contexto.

## Problemas encontrados

En esta sección detallamos los principales desafíos que surgieron durante el desarrollo del proyecto y las estrategias implementadas para resolverlos.

### Recursos y entorno de ejecución

Comenzamos utilizando Google Colab en el proyecto debido a sus ventajas,  como la accesibilidad, colaboración en tiempo real y disponibilidad de GPU gratuitas. Continuamos usando esta plataforma en cada etapa, incluyendo el *fine tuning*. Sin embargo, a medida que el proyecto avanzaba, los recursos ofrecidos por este entorno comenzaron a limitarnos, lo que nos hizo recurrir a las técnicas de cuantización y optimización del rendimiento mencionadas anteriormente.

### Elección del tipo de preguntas a generar

En primera instancia, probamos el desempeño de diferentes modelos en la terea de generar preguntas en base a un artículo. No obstante, la comparación entre las preguntas generadas cada vez fue más difícil de hacer, lo que nos hizo pensar que nuestra forma de categorizar preguntas como "buenas" o "malas" carecía de certeza, provocando incertidumbre al momento de decidir si la calidad de las preguntas generadas mejoraba. Para solucionarlo, definimos una serie de pautas acerca de la características que deseamos que las preguntas generadas por el modelo posean:
* *Tipos de preguntas:* los tipos de preguntas que se desean generar son de respuesta corta y sentencias a determinar en verdaderas o falsas. Además, se considerará que la respuestas a dichas preguntas estén presentes dentro del artículo proporcionado.
* *No repetición:* deseamos diversidad de preguntas, por lo tanto, dos de la misma lista no deben ser ni idénticas ni demasiado similares. A continuación, dejamos un ejemplo de preguntas similares que queremos evitar.
  ```
   Pregunta 1: ¿Qué hizo en el Siglo 19?
   Pregunta 2: ¿Qué hizo en el Siglo 20?
  ```
* *Naturalidad y fluidez:* las preguntas generadas deben evitar una estructura mecanizada, minimizando repeticiones innecesarias y mostrando una formulación de preguntas similiar a las realizadas por una persona.
* *Completitud:* priorizaremos la relevancia de las preguntas pero tendremos en cuenta que sean abarcativas a los temas tratados en el articulo.
* *Orientadas al refuerzo de la comprensión:* las preguntas buscarán evaluar a quien las responda en su habilidad para identificar ideas principales y detalles específicos (por ejemplo: fechas).

### Elección del modelo

Definir el modelo y la versión a usar del mismo nos llevo más tiempo del que esperábamos, ya que tuvimos que investigar y probar diferentes posibilidades.

Debíamos buscar modelos que tengan soporte para la generación de texto en español. En dicha búsqueda, encontramos a la familia de modelos LLaMa en su versión 3.1 que son multilingües [16]. Para comprender la importancia de este punto, en la siguiente subsección mostraremos una comparativa sencilla entre la capacidad de generación de texto en español del modelo `meta-llama/Llama-2-13b-hf`, que no tiene soporte oficial para el idioma, y `meta-llama/Llama-3.1-8B`.

Además, como mencionamos en la sección *system prompt*, terminamos eligiendo el modelo `meta-llama/Llama-3.1-8B-Instruct` ya que nos otorgaba más facilidades para solicitarle instrucciones al modelo.

#### Comparativa: soporte para la generación de texto en español

La comparativa constó en solicitarle a ambos modelos completar una sentencia que comenzaba con `"El perro es"`, con los mismos parámetros de generación, entre los cuales incluimos un número para `max_new_tokens` para así evitar secuencias generadas que sean muy largas. La siguiente imagen es un ejemplo de los textos generados por ambos modelos.

![Compartiva](https://i.imgur.com/UsBXOZ6.png)

Como ya hemos mencionado a lo largo del artículo, la elección del modelo  `"meta-llama/Llama-3.1-8B-Instruct"` fue clave para el *prompting*, debido a que puede interpretar instrucciones y usar roles en su entrada para adaptar su salida [6].

### Reducir el dominio

Los artículos presentes en Wikipedia pueden tratarse de diversos temas, asimismo una pregunta puede ser menos relevante en un artículo que en otro. Es por eso que decidimos reducir el dominio de los artículos a solo biografías, sobre los cuales evaluaremos las preguntas generadas. De esta manera, sería más fácil evaluar la calidad de las preguntas y detectar las limitaciones del modelo.

### Respuestas no solicitadas junto con las preguntas generadas

La tarea que le solicitamos al modelo siempre fue generar preguntas a partir de un determinado contexto. Uno de los problemas que surgieron con esta solicitud fue que el modelo también generaba las respuestas de dichas preguntas, inclusive si le especificábamos explícitamente que no lo haga, a través del rol `system`. A este problema pudimos solucionarlo parcialmente, mediante la técnica de *few shot prompting* en conjunto con una mejora en la redacción de la solicitud suministrada realizando una más corta y directa. Sin embargo, la solución definitiva a este problema fue el *fine tuning*.

### Error en una URL del conjunto de datos

El conjunto de datos que decidimos utilizar está presente como `Dataset` en Hugging Face [14], para facilitar su uso mediante su API. No obstante, el módulo encargado de descargarlo llamado `SQAC.py` posee un error en una URL que nos lleva a una página que no existe y lo cual no permite obtenerlo mediante el comando `load_dataset("PlanTL-GOB-ES/SQAC")`. Para solucionarlo, debimos descargar los archivos del conjunto de datos desde la página de HF y cambiar el valor de la variable `_URL` del archivo `SQAC.py` por `"https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/"`.

Este error lleva tiempo ahí, inclusive un usuario solicitó un *pull request* con la solución, sino que aún no ha sido aceptada [17].

### Familia de modelos LLama3 y los tokens de relleno

El tokenizador de la familia de modelos LLaMa, no poseen un `pad_token` definido, lo que es un problema para hacer inferencia, y también, para aplicar *fine tuning* sobre ellos. Hemos investigado mucho al respecto, y es un hecho, la falta de información oficial sobre este tema provoca una gran confusión entre los usuarios [18][19]. La solución que abordamos para este problema fue la que habitualmente se utiliza, definir el `pad_token` igual que el `eos_token` y mantener un tamaño de *batch* de 1 durante el entrenamiento para evitar problemas debido al token de relleno.

### Fine tuning

Adoptamos como primer enfoque implementar la técnica de *fine tuning* sobre 1000 preguntas repartidas en 840 contextos diferentes del conjunto de datos SQAC. No obstante, en el entrenamiento del modelo pudimos notar que algo andaba mal debido a que el *training loss* no disminuía, sino que oscilaba siempre entre los mismos valores, como se puede observar en el siguiente gráfico:

![Training loss sin cambios](https://i.imgur.com/wPDLTNc.pngg)

Al finalizar el entrenamiento y evaluar el modelo, observamos que la generación de texto carecía de coherencia y naturalidad. Además, el modelo tendía a repetir las pocas preguntas que lograba formular y no incluía un token especial de terminación, sino que entraba en un bucle generando contenido repetitivo (lo que suele ser un problema entre los usuarios [20]).

Nuestra hipótesis fue que el modelo no lograba captar el patrón en los datos suministrados para lograr aprender de ellos. En una primera instancia, pensamos que era porque había demasiados tipos de artículos distintos en los fragmentos del conjunto de datos, por lo que probamos entrenar el modelo nuevamente, pero esta vez, filtrando aquellos contextos que provengan de la sección "Biografía" de algún artículo, y entrenando el modelo con conversaciones formuladas a partir de las preguntas asociadas a tales contextos.

Cabe aclarar que un mismo contexto puede poseer varias preguntas asociadas, lo que provoca que se repita varias veces en el conjunto de datos (tantas veces como preguntas asociadas tenga), entonces fue necesario agrupar las preguntas por contexto para generar las conversaciones antes mencionadas.

![Comparacion training loss 1](https://i.imgur.com/w6N63VM.png)

Como podemos ver en la imagen anterior, este nuevo enfoque hizo que el *training loss* disminuyera. Sin embargo, los problemas de generación seguían sin ser solucionados. Por lo tanto, la causa del problema era otra y efectivamente, revisando la documentación de las funciones que habíamos usado para entrenar el modelo notamos que `SFTTrainer` tenía un formato específico para recibir el conjunto de datos de entrenamiento sin necesidad de ser pre-procesado directamente antes, que es lo estábamos haciendo mediante el `tokenizer` del modelo [21]. Entonces, elegimos el formato conversacional el cual tiene la siguiente estructura:

```python
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "..."}]}
```
Con el cual, como podemos ver en la imagen a continuación obtuvimos una mayor disminución del *training loss* y además, el modelo entrenado genera un token especial para indicar cuando ha terminado.

![Comparacion training loss 2](https://i.imgur.com/ajH9UgV.png)


###  Evaluación del progreso
Hasta ahora se realizó una evaluación anecdótica manual comparando los diferentes resultados obtenidos al cambiar las entradas y las técnicas utilizadas. Se planea para un futuro investigar alternativas de evaluación.

****************************************

### Referencias

1. https://urjconline.atavist.com/2023/06/29/pregunta-tipo-respuesta-corta/.
2. https://www.llama.com/docs/how-to-guides/quantization/.
3. https://www.llama.com/docs/how-to-guides/prompting.
4. https://www.promptingguide.ai/techniques/zeroshot.
5. https://huggingface.co/blog/llama31#how-to-prompt-llama-31.
6.  https://huggingface.co/blog/llama31#using-hugging-face-transformers.
7.   https://promptengineering.org/system-prompts-in-large-language-models/
8.   https://www.turing.com/resources/finetuning-large-language-models.
9.   https://toloka.ai/blog/base-llm-vs-instruction-tuned-llm/.
10.   https://promptengineering.org/system-prompts-in-large-language-models/.
11.   https://www.datacamp.com/tutorial/quantization-for-large-language-models?utm_source=chatgpt.com.
12.   https://llamaimodel.com/requirements/.
13.   https://huggingface.co/blog/4bit-transformers-bitsandbytes.
14.   https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC.
15.   https://ai.meta.com/blog/meta-llama-3-1/.
16.   https://portal.odesia.uned.es/dataset/sqac-es.
17.   https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/discussions/4
18.   https://discuss.huggingface.co/t/llama-pad-token/48001.
19.   https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418/6.
20.   https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/60.
21.   https://huggingface.co/docs/trl/v0.11.1/en/sft_trainer#dataset-format-support.
22.   https://huggingface.co/docs/transformers/v4.32.1/llm_tutorial.
