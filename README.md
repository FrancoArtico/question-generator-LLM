# Generación de preguntas de lectura comprensiva en español utilizando LLM

#### Resumen
Este proyecto presenta una prueba de concepto para un generador de preguntas de lectura comprensiva en español, que se enfocará principalmente en dos tipos de preguntas: de respuesta corta [^1] y afirmaciones a evaluar como verdaderas o falsas. Para ello, utilizaremos el modelo de lenguaje `"Llama-3.1-8B-Instruct"` aplicando distintas técnicas y evaluando sus posibles mejorías en el desempeño. A lo largo de este proyecto trabajaremos principalmente sobre artículos de Wikipedia. Consideramos este estudio de interés dado que la viabilidad de un sistema con estas características podría reducir la carga horaria de profesores y reforzar el aprendizaje de los estudiantes sobre los textos.

**Autores:** *Franco Artico* y *Nain Cadro*.

## Hipótesis, objetivos iniciales y el estado final alcanzado

Nuestra **hipótesis inicial** planteaba que el modelo seleccionado presentaría dificultades en la generación de preguntas de lectura comprensiva a partir de un texto en español, mostrando errores sintácticos o generando preguntas que no se corresponden con las estipuladas en el proyecto. Para poner a prueba nuestra hipótesis, primero investigamos y seleccionamos un modelo para evaluarlo. Además, fue imprescindible definir el tipo de preguntas que deseábamos generar y ciertos criterios para también evaluarlas.

Con el modelo seleccionado (`meta-llama/Meta-Llama-3-8B-Instruct`) realizamos diversas pruebas que, efectivamente, mostraron deficiencias en la tarea solicitada, las cuales pudieron ser identificadas y abordadas mediante las técnicas establecidas en los objetivos.

Durante el desarrollo del proyecto ajustamos nuestro enfoque, reduciendo el dominio del problema a biografías o fragmentos de ellas. Este cambio nos permitió identificar más fácilmente las dificultades en la generación de preguntas, en comparación con trabajar con la extensa variedad de artículos disponibles en Wikipedia.

De esta manera, la siguiente lista muestra cómo quedaron constituidos los objetivos que nos propusimos alcanzar a lo largo del proyecto:
1. Investigar acerca de los modelos de lenguaje entrenados para generar texto en español.
2. Seleccionar el tipo de texto específico sobre el cual el modelo realizará preguntas.
3. Definir criterios de evaluación para las preguntas generadas por el modelo de acuerdo con el objetivo del proyecto.
4. Detectar limitaciones del modelo elegido para la generación de preguntas en español.
5. Aplicar *zero-shot prompting* y evaluar los resultados obtenidos.
6. Aplicar *few shot prompting* y evaluar los resultados obtenidos.
7. Elegir un conjunto de datos apropiado para realizar el *fine tuning* del modelo.
8. Efectuar *fine tuning* sobre el modelo y buscar mejores resultados respecto a los obtenidos mediante técnicas de *prompting*.
9. Determinar las mejoras obtenidas en cada enfoque y establecer conclusiones acerca de las mismas.

A través de estos objetivos logramos abordar las limitaciones detectadas del modelo y aplicar diferentes técnicas para mejorar la calidad de las preguntas generadas. Los resultados obtenidos mediante estas técnicas serán analizados con más detalle en las conclusiones del informe. 

## Técnicas relevantes

Existen diferentes técnicas para lograr que un modelo realice de forma correcta una tarea específica. En este proyecto se abordaron las siguientes: *zero-shot learning*, *few-shot learning*, *system prompts*, cuantización y *fine tuning*.

### Zero-shot learning
*Zero-shot learning* (sin ejemplos) es una técnica que consiste en interactuar con el modelo proporcionando la instrucción a realizar sin incluir ejemplos ni demostraciones de cómo queremos que sea realizada [^2].

Esta técnica la utilizamos inicialmente buscando descubrir la clase de preguntas generadas por **los modelos base** de la familia Llama 3.1 que no poseen un entrenamiento específico en esta tarea.

Un **modelo base** no sigue instrucciones. Fue entrenado sobre una gran cantidad de datos para adquirir conocimientos básicos acerca del lenguaje, conocimiento general y contextos, por lo que es útil en situaciones donde debe dar respuestas de conocimiento general o conversaciones casuales, siempre produciendo el siguiente *token* seleccionado de acuerdo a la técnica de generación. Por lo tanto, para obtener lo que deseamos en este proyecto, debemos solicitarlo de forma diferente, agregando más indicaciones o guías [^3].

A continuación, presentamos un ejemplo de entrada utilizando esta técnica junto con las preguntas obtenidas usando el modelo `meta-llama/Llama-3.1-8B` y ciertas obvservaciones al respecto.

#### Ejemplo entrada del modelo
presentamos un ejemplo de entrada utilizando esta técnica junto con las preguntas obtenidas usando el modelo `meta-llama/Llama-3.1-8B` y ciertas obvservaciones al respecto.

```python
prompt = article[0] + article[1] + "\n\n En base al texto anterior, responder las siguientes 5 preguntas:"
```
donde `article[0]` y `article[1]` son el primer y segundo párrafo respectivamente de la biografía de Bill Gates de Wikipedia. De esta manera, incentivamos al modelo a completar nuestra entrada generando preguntas sobre el artículo.

#### Salida producida
>  ...
>  
>  En base al texto anterior, responder las siguientes 5 preguntas: 
> 1. ¿Cuál es la edad de Bill Gates?
> 2. ¿En qué año nació Bill Gates?
> 3. ¿Cuál es el nombre de su hijo?
> 4. ¿Cuál es la edad de su hijo?
> 5. ¿Cuál es el nombre de su hija?
> 6. ¿Cual es la edad de su hija?
> 7. 8 9 10 11 12 13 14 ...

En esta salida, observamos que el modelo incumplió con la cantidad de preguntas solicitadas, es más, las pocas generadas son muy similares entre sí y encima termina produciendo números consecutivos en lugar de preguntas. Es más, las respuestas a algunas las preguntas producidas ni siquiera se encuentran dentro del texto proporcionado.

### Few-shot learning

*Few-shot learning* es una técnica que consiste en pasarle al modelo, junto con la instrucción de la tarea, unos pocos ejemplos que sirvan como guía para aprender el patrón de los resultados que deseamos obtener [^4].

La idea de utilizar esta técnica en nuestro proyecto surge, principalmente, al darnos cuenta que los modelos naturalmente no generan preguntas de verdadero o falso, como pudimos ver en el ejemplo de la sección *zero-shot*. Entonces, necesitábamos proporcionarles ejemplos para incentivarlos a generarlas. 

#### Ejemplo entrada del modelo


```python
prompt = (
    f"{article}\n\n"
    "En base al texto anterior, responde las siguientes 10 preguntas:\n"
    "1. Bill nació en Palo Alto, California, Estados Unidos. ¿Verdadero o falso?\n"
    "2. Bill Gates se graduó de Harvard con honores. ¿Verdadero o falso?\n"
    "3. Bill logró venderle MS-DOS a IBM. ¿Verdadero o falso?\n"
    "4. Gates nunca tuvo hijos. ¿Verdadero o falso?\n"
)
```

En este caso mediante `article`, le proporcionamos al modelo el artículo completo. 

#### Salida producida
> En base al texto anterior, responde las siguientes 10 preguntas:
> 1. Bill nació en Palo Alto, California, Estados Unidos. ¿Verdadero o falso?
> 2. Bill Gates se graduó de Harvard con honores. ¿Verdadero o falso?
> 3. Bill logró venderle MS-DOS a IBM. ¿Verdadero o falso?
> 4. Gates nunca tuvo hijos. ¿Verdadero o falso?
> 5. El nombre de la empresa de software de Bill Gates es Microsoft. ¿Verdadero o falso?
> 6. Bill Gates se casó con Melinda French. ¿Verdadero o falso?
> 7. Bill Gates es el segundo hombre más rico del mundo. ¿Verdadero o falso?
> 8. Bill Gates es el creador de Microsoft. ¿Verdadero o falso?
> 9. Bill Gates nació en Seattle, Washington, Estados Unidos. ¿Verdadero o falso?
> 10. Bill Gates se graduó de Harvard con honores. ¿Verdadero o falso?
> 
> Respuestas
> 
> 1. F
> 2. V
> 3. V
> 4. F
> 5. V
> 6. V
> 7. F
> 8. V
> 9. V
> 10. V
> 
> Referencias:
> 1. 1 2 3  «Bill y Melinda Gates, la pareja más rica del mundo, anuncian su divorcio». BBC News Mundo. Consultado el 4 de mayo de 2021.
> 2. ↑  «Bill Gates, el hombre más rico del mundo». BBC News Mundo. Consultado el 13 de mayo de 2023...

Este resultado está lejos de ser ideal pero, como mencionamos, logramos que el modelo genere preguntas de verdadero o falso simplemente añadiendo algunos ejemplos, que era lo que buscábamos mediante este enfoque. En este punto, todavía quedaban varios aspectos a mejorar como: evitar incluir respuestas junto a las preguntas, falta de variedad en el tipo de preguntas y que no siga generando luego de que completó la instrucción solicitada. Veremos como estos aspectos se fueron mejorando a lo largo del informe.


### System prompts

*System prompts* son un conjunto de instrucciones, guías e información acerca del contexto que se le provee al modelo antes de interactuar con las consultas del usuario [^3].

El uso de modelos base para la generación de preguntas presenta varias limitaciones. En primer lugar, no permite especificar de manera directa la cantidad y el tipo de preguntas a generar. Además, requieren estrategias más avanzadas de ingeniería de *prompts* para guiar de forma adecuada la generación de las preguntas, lo que implica un mayor esfuerzo en el diseño de instrucciones óptimas. Por estos motivos, se decidió optar por un modelo instruido como `meta-llama/Meta-Llama-3.1-8B-Instruct`, lo que también tendrá un impacto positivo en el *fine tuning*, ya que nos permite enfocarnos en mejorar el desempeño en la tarea específica, sin la necesidad de enseñarle al modelo a interpetar las instrucciones desde cero.

Este cambio de modelo, nos permitió aplicar esta técnica en nuestro proyecto. Dado que `Meta-Llama-3.1-8B-Instruct`  cuenta con ciertos roles, entre los cuales destacamos *system*, *user* y *assistant*. El rol *system* sirve para establecer el contexto de una conversación o proveerle al modelo información necesaria para ayudarle a responder efectivamente y es el que juega un papel fundamental en esta técnica, mientras que el rol *user* y *system* representan la entrada del usuario y la respuesta del modelo respectivamente [^5].

#### Ejemplo entrada del modelo

En el siguiente fragmento de código mostramos cómo definimos los roles `system`  y `user` para realizar consultas al modelo.

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

#### Salida producida
>Aquí te presento 10 preguntas basadas en el texto, de las cuales 5 son de tipo verdadero o falso:
> 
> 1. ¿Qué empresa cofundó Bill Gates junto con Paul Allen?
> 2. ¿Cuál fue el cargo de Bill Gates en Microsoft antes de mayo de 2014?
> 3. ¿Cuál fue el valor de la fortuna de Bill Gates en 2023, según la revista Forbes?
> 4. (Verdadero o falso) Bill Gates se casó con Melinda French en 1986.
> 5. ¿Cuál fue el nombre del laboratorio de computación donado por Bill Gates al Massachusetts Institute of Technology (MIT) en 1999?
> 6. ¿Qué evento deportivo de caridad participó Bill Gates en 2017?
> 7. (Verdadero o falso) Bill Gates abandonó su cargo en Microsoft en 2008 para dedicarse a su fundación.
> 8. ¿Cuál fue el valor de la donación de Bill Gates a la Fundación Bill y Melinda Gates en julio de 2022?
> 9. (Verdadero o falso) Bill Gates es un creyente ferviente y ha sido un líder religioso.
> 10. ¿Cuál fue el nombre de la serie documental lanzada por Netflix en 2019 sobre la vida de Bill Gates?

Como podemos contemplar, este enfoque por sí solo no era suficiente para tener una salida adecuada. Había varios puntos débiles para seguir trabajando como: el número generado de preguntas de tipo verdadero o falso no era el especificado en la consulta, temas importantes de la biografía de Bill Gates no son preguntados y todas las preguntas son de la forma ""¿Qué" o "¿Cuál".

Es por eso que, optamos por combinar esta técnica con *few shot* y los resultados son analizados en la conclusión del informe.

### Cuantización
La cuantización es una técnica que nos permite comprimir modelos extensos de lenguaje representando sus pesos y activadores, los cuales están típicamente en números flotantes de 32-bits de precisión, con valores de menor precisión, usualmente 8-bits o 4-bits. La reducción en el número de bits provoca una gran disminución en el tamaño del modelo, por lo que consumen menos memoria, requieren menos espacio de guardado y son más rápidos de aplicarle *fine tuning* [^6][^7].

![imagen](https://i.imgur.com/3WSwSJF.png)

El modelo `"meta-llama/Llama-3.1-8B-Instruct"` requiere disponer de al menos 16 GB de RAM [^8]. Sin embargo, dado que nuestro entorno de ejecución en Google Colab solo ofrecía 15 GB de RAM, se hizo necesario aplicar cuantización para reducir el modelo. Para ello, implementamos la técnica NF4 (*4-bit Normal Float*), la cual cuantifica los pesos a 4 bits, permitiéndonos ejecutar y evaluar distintos modelos (como Llama2-7b, Llama3.1-8B y Llama3.1-8B-Instruct) pese a las limitaciones de recursos del entorno.

Además, un artículo de Hugging Face [^9], nos permitió saber antes de entrenar nuestro modelo que sería posible hacerlo en este entorno limitado usando GC (*gradient checkpointing*) y *nested quantization*.

### Fine tuning

*Fine tuning* es el proceso de ajustar los parámetros de un modelo extenso de lenguaje pre-entrenado para una tarea o dominio específico a través de un conjunto de datos. Esto dado que, los grandes modelos poseen un conocimiento amplio pero carecen de especialización en áreas específicas, por lo que esta técnica ataca ese problema haciendo a los modelos más acertados y efectivos para ciertos dominios o aplicaciones [^10][^11].

*Supervised Fine Tuning* es un método para mejorar y personalizar modelos pre-entrenados, que consiste en volver a entrenarlos en un conjunto de datos etiquetado, compuesto por instrucciones y las respuestas deseadas. Las tres técnicas más populares para aplicar SFT son *full fine-tuning*, *LoRA* y *QLoRA*.

![imagen](https://i.imgur.com/IG9s8Yk.png)

*QLoRA (Quantization-aware Low-Rank Adaptation)* es una técnica que en lugar de entrenar el modelo entero, congela los pesos e introduce pequeños adaptadores denominados *low-rank matrices* lo que  permite reducir considerablemente el número de parámetros a entrenar, el uso de la memoria y el tiempo de entrenamiento respecto al *full fine-tuning*. Esta técnica es ideal para entornos donde la memoria GPU es limitada y por eso fue elegida para entrenar nuestro modelo en Colab. La siguiente imagen ilustra la reducción de parámetros entrenables que esta técnica realizó en nuestro caso.

![Parámetros reducidos](https://i.imgur.com/vqrHSB7.png)

El conjunto de datos que seleccionamos para esta técnica fue *Spanish Question Answering Corpus* (SQAC) [^12], que se compone de preguntas-respuestas extractivas en español. Sin embargo, lo consideramos adecuado para la tarea de este proyecto, ya que posee contextos extraídos de artículos de Wikipedia, noticias de WikiNews y la sección española de AnCora corpus que es una mezcla de diferentes fuentes de noticias y literatura. Además, no posee preguntas sin respuestas y tanto las preguntas como las respuestas fueron anotadas por hablantes nativos de español con estudios universitarios [^13]. 

Ahora bien, a este conjunto de datos fue necesario realizarle un pre-procesamiento donde solamente nos quedamos con contextos provenientes de la sección "Biografías" de artículos de Wikipedia. Posterior a eso, tomamos las columnas `context` y `question` para, a partir de ellas, armar una sola (`text`) representando una conversación entre un usuario que provee un contexto y solicita una cierta cantidad de preguntas, y un asistente que genera las preguntas a partir de ese contexto.

A continuación, mostramos el formato utilizado en las conversaciones suministradas al modelo:

```python
[
  {
    "role": "user",
    "content": "Genera exactamente 3 preguntas de lectura comprensiva, sin incluir las respuestas ni opciones, 
      sobre el siguiente texto:\n\nBiografía\n Andrade nació en São Paulo, 
      ciudad en la que vivió durante prácticamente toda su vida, en el número 320 de la Rua Aurora, 
      donde residían sus padres, Carlos Augusto de Moraes Andrade y Maria Luísa Leite Moraes Andrade. 
      En su infancia fue considerado un niño prodigio como pianista. Al mismo tiempo, estudiaba historia, 
      arte y, especialmente, poesía. Tenía un sólido dominio de la lengua francesa, y leyó durante su infancia a 
      Rimbaud y a los principales poetas simbolistas franceses. Aunque escribió poesía desde su más tierna infancia 
      (su primer poema data de 1904), su primera vocación fue la música, 
      y en 1911 se matriculó en el Conservatorio de São Paulo."
  },
  {
    "role": "assistant",
    "content": "Aquí tienes algunas preguntas sobre el texto proporcionado:
      1. ¿Dónde pasó gran parte de su vida Mário de Andrade?
      2. ¿En qué calle vivían los padres de Andrade?
      3. ¿Cuál fue el idioma extranjero que hablaba Andrade fluidamente?"
  }
]

```

#### Ejemplo entrada del modelo

En esta sección presentaremos un ejemplo de una solicitud de generación al modelo entrenado mediante la técnica previamente mencionada, con la respectiva salida del mismo.

```python
messages = [
    {
        "role": "system",
        "content": (
            "Eres un generador de preguntas de lectura comprensiva. "
            "Tu tarea es crear preguntas basadas en un texto dado, "
            "sin incluir respuestas ni opciones. Solo debes generar preguntas, "
            "incluyendo preguntas de tipo verdadero o falso."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Genera exactamente 10 preguntas, sobre el siguiente texto:\n{article}"
        ),
    },
]
```

#### Salida producida
> 1. ¿En qué año nació Bill Gates? 
> 2. ¿Cómo se conoce comúnmente a Bill Gates en su familia?
> 3. ¿Con quién fundó Microsoft?
> 4. ¿Cuál fue el primer sistema operativo que desarrolló Gates y vendió a IBM?
> 5. ¿A quién obtuvo Microsoft la tecnología del entorno gráfico y del ratón?
> 6. ¿Cuál fue el primer ordenador que utilizaba un entorno gráfico y un ratón?
> 7. ¿Cuál es la fortuna de Bill Gates según la revista Forbes?
> 8. ¿Cuál fue la primera donación personal recibida por Gates a la Universidad de Massachusetts Institute of Technology?
> 9. ¿Con quién participó Gates en el Match for Africa 4 en 2017?
> 10. ¿Cuál es la empresa que ha adquirido Bill Gates en septiembre de 2019 por 30 millones de dólares?

## Librerías exploradas y elecciones

### Transformers

![Transformers](https://i.imgur.com/jSFvf9J.png)

Transformers es una librería mantenida por Hugging Face y su comunidad, la cual proporciona APIs para descargar y entrenar modelos pre-entrenados. El uso de estos reduce los costos de cómputo y de tiempo que conllevarían entrenar un modelo desde cero [^14].

Optamos por esta librería tras elegir el modelo, la cual era mencionada en la página de Meta donde lo presentaban [^15]. Su elección se debió a diversos motivos, entre ellos podemos destacar:
* **Fácil acceso a modelos pre-entrenados**, permite acceder a diversos modelos de forma simple, lo que aceleraría el desarrollo de nuestra prueba de concepto. 
* **Documentación** [^16], tutoriales y una comunidad activa en sus foros, facilitando el aprendizaje y la resolución de problemas. 
* **APIs** para distintos niveles de conocimiento, permitiendo adaptar el desarrollo a medida que aprendíamos. 
* **Sencillez**, ya que posibilita realizar inferencia, técnicas de *prompting* y ajustar modelos pre-entrenados en unas cuantas líneas de código, ahorrando tiempo y recursos [^17].

### PyTorch

PyTorch es una librería optimizada de tensores para aprendizaje profundo usando GPUs y CPUs [^18]. Los tensores, matemáticamente, son matrices multi-dimensionales y en este contexto, representan información aparentemente no numérica, como pueden ser textos o imágenes.

En el aprendizaje profundo se manejan grandes volúmenes de datos, esto sumado  a que íbamos a estar trabajando en un entorno con GPU, nos llevó a seleccionar PyTorch para manejar tensores, ya que permite realizar cálculos de estos tanto en CPU como en GPU (contrario a los *ndarrays* de NumPy) [^19], y además es compatible con Transformers de Hugging Face para la familia de modelos Llama 3 [^20].

### Bitsandbytes

Los modelos extensos de lenguaje consumen muchos recursos para ser ejecutados y entrenados, por lo tanto, el acceso a ellos es un desafío para los usuarios [^9]. Bitsandbytes es una librería que permite la cuantización en k-bits usando PyTorch, haciendo a los modelos extensos de lenguaje más accesibles [^21].

La elección de esta librería se debió principalmente a que permitía la cuantización en 4-bits usando NF4 y, tal como se menciona en su documentación, era ideal para aplicar la técnica de QLoRA en el *fine tuning* del modelo.

### Wikipedia-API

`Wikipedia-API` es un *wrapper* sencillo de usar para las APIs de Wikipedia, que permite extraer textos, secciones, links, categorías, traducciones y demás [^22]. Habíamos explorado otro *wrapper* incluso más simple [^23], pero al necesitar procesar los artículos antes de pasárselos al modelo para hacer inferencia, comenzamos a tener dificultades para eliminar secciones no relevantes.

Usamos el *wrapper* seleccionado para obtener los artículos sobre los cuales luego le solicitábamos al modelo realizar las preguntas.

### Datasets y Pandas
Datasets es una librería para acceder y compartir fácilmente conjuntos de datos para diversas tareas, entre ellas el procesamiento de lenguaje natural [^24].

Elegimos utilizar esta librería debido a que nos proveía acceso al conjunto de datos seleccionado, es fácil de utilizar y provee métodos poderosos para el procesamiento de datos (como `map` y `filter`). Además, está diseñada para manejar de forma eficiente grandes conjuntos de datos, evitando cargarlos directamente en la memoria RAM usando Apache Arrow [^25].

Sin embargo, no fue posible sacarle provecho al máximo a esta elección. En un momento tomamos la decisión de convertir nuestro conjunto de datos en un DataFrame de Pandas [^26], el motivo fue que nos facilitaba agrupar las preguntas por contextos mediante el método`groupby`. No obstante, esto no representó ninguna dificultad, ya que ambas librerías cuentan con métodos para convertir entre DataFrames y Datasets.

### TRL

TRL provee herramientas para entrenar modelos de lenguaje que posean la arquitectura de Transformers mediante aprendizaje de refuerzo [^27].

La elección de esta libreria fue crucial para aplicar *supervised fine tuning* [^28]. En nuestro caso, los datos etiquetados estaban conformados por preguntas del conjunto de datos SQAC, y TRL mediante la clase `SFTTrainer` nos facilitó poner esta técnica en práctica en nuestro proyecto.

Otros motivos de su elección fueron su integración con la librería Transformers de Hugging Face, la amplia documentación que posee y el soporte con PEFT (librería cuya importancia detallaremos en la próxima sección).

### PEFT
PEFT es una librería que nos permite adaptar de manera eficiente modelos extensos de lenguaje sin entrenar todo los párametros del modelo (*full fine-tuning*). En su lugar, los métodos de PEFT solo entrenan una pequeña cantidad de parámetros extra, reduciendo los costos de entrenamiento en términos de recursos computacionales y almacenamiento, con un rendimiento comparable a haber entrenado el modelo completo [^29].

Optamos por esta librería debido a su integración con Transformers y a las herramientas que proporciona para configurar y aplicar QLoRA de manera sencilla.

### Wandb

Es una librería de la plataforma *Weights & Biases* que ofrece herramientas para entrenar modelos, afinarlos y aprovechar modelos fundamentales [^30].

Esta plataforma nos permitó llevar un seguimiento de los hiper-parámetros  utilizados para entrenar el modelo junto con un monitoreo en tiempo real del comportamiento del mismo durante el entrenamiento. Asimismo, ofreció un registro de varias métricas del modelo como el *training loss* mediante gráficos muy útiles. Todo esto, además, facilitó la comparación de diferentes entrenamientos realizados.

### Unsloth

Unsloth es una herramienta que promete inferencia y entrenar modelos extensos de lenguaje más rápido y con menos recursos, sin pérdida de *accuracy* [^31]. En nuestro caso, intentamos primero usarlo para inferir. Como resultado, obtuvimos una velocidad de inferencia mayor comparada a la realizada sin el uso de esta herramienta, pero notamos una degradación en la calidad de respuesta utilizando contextos largos (alrededor de 15.000 tokens), llegando incluso a generar resúmenes en lugar de preguntas sobre el texto solicitado, por lo que optamos por no utilizarla.

Nuestra hipótesis sobre esta degradación se basa en la forma en la que Unsloth maneja los contextos a través de *RoPE Scaling* [^32]. Consideramos que, al procesar contextos de este tamaño, el modelo podría perder la capacidad de retener la instrucción solicitada, tendiendo en su lugar, a realizar la instrucción más común asociada con dicha longitud de contexto.

## Bases de código exploradas

Dado a que era nuestra primera vez trabajando en un proyecto sobre el procesamiento de lenguaje natural, fue necesario explorar diferentes bases de código para aprender las técnicas más utilizadas, las diferentes partes en el desarrollo y para mantener nuestro código simple y fácil de leer pero modernizado, ya que varias de las librerías utilizadas en el proyecto se actualizan de forma regular.



| Nombre                                                                                  | Repositorio                                                                                                      | Propósito                                                                          | Aspectos clave                                                                              | Motivo de análisis                                                                                                                              |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
|    🤗 **Tokenizers - Hugging Face**                                                                                       |                                                                                             [huggingface/tokenizers](https://github.com/huggingface/tokenizers)                     | Pre-procesameinto de datos, inferencia y entrenamiento.                                                                                    |     Buena documentación y muchos ejemplos.                                                                                        | Ofrece una implementación optimizada para la tokenización rápida y eficiente, compatible con los modelos del Hub. |
| 🤗 **Datasets - Hugging Face**                                                          | [huggingface/datasets](https://github.com/huggingface/datasets)                                                  | Descarga y preprocesamiento de datos.                                              | Carga de conjunto de datos públicos de forma sencilla y pre-procesado eficiente.            | Ofrece código para descargar conjuntos de datos y diferentes configuraciones para hacerlo.                                                      |
| 🤗 **TRL - Hugging Face**                                                               | [huggingface/trl](https://github.com/huggingface/trl)                                                            | *Fine tuning* usando aprendizaje supervisado.                                      | Soporte con PEFT, buena documentación, eficiencia y `Trainers`.                             | Brinda implementaciones de entrenamientos, ejemplos de formatos de conjunto de datos utilizados para entrenar y ejemplos de configuración de parámetros de entrenamiento. |
| 🤗 **NLP Course - Hugging Face**                                                        | [huggingface/course](https://github.com/huggingface/course)                                                      | Inferencia de los distintos modelos probados y comprensión de conceptos claves.    | Aprendizaje guiado sobre el ecosistema de HuggingFace y conceptos clave de NLP.             | Instruirnos en el uso de `Transformers`, `Dataset` y `Tokenizers`.                                                                              |
| 🤗 **Transformers - Hugging Face**                                                      | [huggingface/transformers](https://github.com/huggingface/transformers)                                          | Implementación de modelos LLM.                                                     | Compatibilidad con modelos Llama y comunidad activa.                                        | Base para hacer inferencia probando modelos mediante `pipeline`.                                                                                |
| 📖 **LLM Course**                                                                       | [mlabonne/llm-course](https://github.com/mlabonne/llm-course)                                                    | *Fine tuning* y pre-procesamiento de datos.                                        | Explicaciones detalladas, entrenamiento ultra-eficiente y código fácil de leer.             | Aprender cómo entrenar el modelo de manera eficiente, usando técnicas de aprendizaje supervisado.                                               |
| 🤗 **PEFT - Hugging Face**                                                              | [huggingface/peft](https://github.com/huggingface/peft)                                                          | *Fine-tuning* usando QLoRA y valor de parámetros para la aplicación de la técnica. | *Fine-tuning* eficiente y buena documentación.                                              | Ofrece explicación detallada acerca de los parámetros para aplicar QLoRA y código donde se aplica la técnica.                                   |
| ⚡ **Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA** | [colab/bnb-4bit-training](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) | *Fine tuning* de modelos mediante QLoRA y cuantización en NF4.                     | Uso del ecosistema de HuggingFace, *nested quantization*, NF4 y *Gradient checkpointing*.   | Ofrece una implementación optimizada usando QLoRA y es compatible con HuggingFace.                                                              |
| 🦥 **Unsloth**                                                                          | [unslothai/unsloth](https://github.com/unslothai/unsloth)                                                        | Optimización para realizar inferencia y *fine tuning*.                             | Optimización de la memoria, compatibilidad con modelos Llama 3.1 y velocidad de inferencia. | Brinda modelos ya cuantizados, código para entrenar y hacer inferencia de modelos de la familia Llama 3, usando la herramienta Unsloth en un entorno de HuggingFace.    |
| 📚 **Wikipedia-API**                                                                    | [project/wikipedia-API](https://pypi.org/project/Wikipedia-API/)                                                 | Obtención de datos de Wikipedia para probar modelos.                               | Simpleza y facilidad para recuperar información de Wikipedia.                               | Ofrece código sobre el uso del *wrapper* mediante casos de uso.                                                                                 |

## Problemas encontrados

En esta sección detallamos los principales desafíos que surgieron durante el desarrollo del proyecto y las estrategias implementadas para resolverlos.

### Recursos y entorno de ejecución

Comenzamos utilizando Google Colab en el proyecto debido a ventajas como la accesibilidad, colaboración en tiempo real y disponibilidad de GPU gratuitas. Continuamos usando esta plataforma en cada etapa, incluyendo el *fine tuning*. Sin embargo, a medida que el proyecto avanzaba, los recursos ofrecidos por este entorno comenzaron a limitarnos, lo que nos hizo recurrir a las técnicas de cuantización y optimización del rendimiento mencionadas anteriormente.

### Elección del tipo de preguntas a generar

En primera instancia, probamos el desempeño de diferentes modelos en la tarea de generar preguntas en base a un artículo. No obstante, la comparación entre las preguntas generadas cada vez fue más difícil de realizar, lo que nos hizo pensar que nuestra forma de categorizar preguntas como "buenas" o "malas" carecía de certeza, provocando incertidumbre al momento de decidir si la calidad de las preguntas generadas mejoraba. Para solucionarlo, definimos una serie de pautas acerca de la características que deseamos que las preguntas generadas por el modelo posean:
* *Tipos de preguntas:* los tipos de preguntas que se desean generar son de respuesta corta y sentencias a determinar en verdaderas o falsas. Además, se considerará que la respuestas a dichas preguntas estén presentes dentro del artículo proporcionado.
    > **✅ Ejemplos de pregunta de repuesta corta:**
    > 1. ¿En que año nació?
    > 2. ¿Cuál fue la fecha y causa de su muerte?
    > 3. ¿Dónde estudió?
    > 4. ¿Qué ocupaciones tuvo?
    > 
    > **✅ Ejemplos de sentencias a determinar como verdaderas o falsas:**
    > 1. Bill Gates es el fundador de la empresa Apple. ¿Verdadero o Falso?
    > 2. Bill Gates es un gran aficionado al tenis. ¿Verdadero o Falso?
* *No repetición:* deseamos diversidad de preguntas, por lo tanto, dos de la misma lista no deben ser ni idénticas ni demasiado similares. 
    > ⛔ **Ejemplos de preguntas repetidas o similares a evitar:**
    > 1. ¿En qué año nació Bill Gates?.
    > 2. ¿Que edad tiene Bill Gates?.
* *Naturalidad y fluidez:* las preguntas generadas deben evitar una estructura mecanizada, minimizando repeticiones innecesarias y mostrando una formulación de preguntas similiar a las realizadas por una persona.
    > ⛔ **Ejemplos de preguntas antinaturales (errores de formulación):**
    >  1. ¿Qué campo de estudio se le considera uno de los más importantes de todos los tiempos? (Corrección: ¿En qué --> ¿Qué).
    >  2.  ¿Qué universidad asistió Gödel a estudiar después de cumplir 18 años? (Corección: ¿A qué --> ¿Qué).
    >  3. ¿Qué fue el resultado de la investigación policial sobre la muerte de Alan Turing en 2012? (Corrección: ¿Qué --> ¿Cuál).
    >  4. ¿En qué universidad asistió Steve Jobs? (Corrección: ¿En qué --> ¿A qué)

    > ⛔ **Ejemplos de preguntas antinaturales (falta de naturalidad):**
    >  1. ¿En qué año Bill Gates abandonó la universidad para trasladarse a Albuquerque y pactar con MITS la cesión del 50% del lenguaje para computadoras Basic?  (posee una estructura mecanizada extraída del texto directamente).
    >  2.  ¿Fue John von Neumann un matemático húngaro-estadounidense que realizó contribuciones fundamentales en física cuántica, análisis funcional, teoría de conjuntos, teoría de juegos, ciencias de la computación, economía, análisis numérico, cibernética, hidrodinámica, estadística y muchos otros campos? (al igual que el primer ejemplo, una pregunta demasiado específica).
    >  3. ¿Verdadero o falso: La reina Isabel II le otorgó a Alan Turing un indulto póstumo en 2013? (no es una forma natural de preguntar acerca de verdaderos o falsos).

    > ⛔ **Ejemplos de preguntas antinaturales (estructura mecanizada):**
    >  1. ¿**Fue** el gobierno británico de David Cameron el que denegó el indulto a Alan Turing en 2012?
    >  2. ¿**Fue** el premio Turing otorgado anualmente por la Association for Computing Machinery para personas destacadas por sus contribuciones técnicas al mundo de la informática?
    > 3. ¿**Fue** el Instituto Alan Turing inaugurado por el UMIST y la Universidad de Mánchester en el verano de 2004?
    > 4. ¿**Fue** la estatua de bronce de Alan Turing esculpida por John W. Mills en la Universidad de Surrey en 2004?
* *Completitud:* priorizaremos la relevancia de las preguntas pero tendremos en cuenta que sean abarcativas a los temas tratados en el artículo.
* *Orientadas al refuerzo de la comprensión:* las preguntas buscarán evaluar a quien las responda en su habilidad para identificar ideas principales y detalles específicos (por ejemplo: fechas).
    > **✅ Ejemplos de pregunta que refuerzan la comprensión:**
    > 1. ¿Cuál fue la fecha de nacimiento de John von Neumann?
    > 2. Cillian Murphy ganó el Óscar como mejor actor por su papel en la película Oppenheimer, ¿Verdadero o Falso? (es una información importante para la biografía de un actor)

### Elección del modelo

Definir el modelo y la versión a usar del mismo nos llevo más tiempo del que esperábamos, ya que tuvimos que investigar y probar diferentes posibilidades.

Debíamos buscar modelos que tengan soporte para la generación de texto en español. En dicha búsqueda, encontramos a la familia de modelos LLaMa en su versión 3.1 que son multilingües [^33]. Para comprender la importancia de este punto, en la siguiente subsección mostraremos una comparativa sencilla entre la capacidad de generación de texto en español del modelo `meta-llama/Llama-2-13b-hf`, que no tiene soporte oficial para el idioma, y `meta-llama/Llama-3.1-8B`.

Además, como mencionamos en la sección *system prompt*, terminamos eligiendo el modelo `meta-llama/Llama-3.1-8B-Instruct` ya que nos otorgaba más facilidades para solicitarle instrucciones.

#### Comparativa: soporte para la generación de texto en español

La comparativa constó en solicitarle a ambos modelos completar una sentencia que comenzaba con `"El perro estaba"`, con los mismos parámetros de generación, entre los cuales incluimos un número para `max_new_tokens` y así evitar secuencias generadas que sean muy largas. La siguiente imagen es un ejemplo de los textos generados por ambos modelos.

![Compartiva](https://i.imgur.com/UsBXOZ6.png)

Como ya hemos mencionado a lo largo del artículo, la elección del modelo  `"meta-llama/Llama-3.1-8B-Instruct"` fue clave para el *prompting*, debido a que puede interpretar instrucciones y usar roles en su entrada para adaptar su salida [^34].

### Reducir el dominio

Los artículos presentes en Wikipedia pueden tratarse de diversos temas, asimismo una pregunta puede ser menos relevante en un artículo que en otro. Es por eso que decidimos reducir el dominio de los artículos a solo biografías, sobre los cuales evaluaremos las preguntas generadas. De esta manera, sería más fácil evaluar la calidad de las preguntas y detectar las limitaciones del modelo.

### Respuestas no solicitadas junto con las preguntas generadas

La tarea que le solicitamos al modelo siempre fue generar preguntas a partir de un determinado contexto. Uno de los problemas que surgieron con esta solicitud fue que el modelo también generaba las respuestas de dichas preguntas, inclusive si le especificábamos explícitamente que no lo hiciera.

#### Ejemplo de respuestas no solicitadas junto a las preguntas generadas
A continuación, mostramos un ejemplo de salida junto a su entrada que nos sucedió usando el modelo instruido.

``` Python
# Entrada suministrada al modelo
prompt = [
    {
        "role": "system",
        "content": (
            "Eres un generador de preguntas de lectura comprensiva. "
            "Tu tarea es generar preguntas, sin incluir las repsuestas " 
            "y ademas algunas de las generadas deben ser de tipo "
            "verdadero o falso"
        ),
    },
    {
        "role": "user",
        "content": (
            "Genera exactamente 10 preguntas, de las cuales 5 deben ser de tipo "
            "verdadero o falso, sin incluir las respuestas, sobre el siguiente texto:\n\n"
            f"{article}"
        ),
    },
```


> **Salida producida por el modelo `meta-llama/Llama-3.1-8B-Instruct`:**
> 1. ¿Cuál fue el año en que Steve Jobs fundó Apple?
> 2. ¿Cuál fue el nombre del primer ordenador personal de Apple que usaba una interfaz gráfica de usuario (GUI) y un ratón?
> 3. ¿Verdad o falsedad: Steve Jobs se convirtió en el mayor accionista individual de Disney después de la fusión de Pixar con la empresa.
> 4. ¿Cuál fue el nombre del software de desarrollo que se basó en el sistema operativo NeXTSTEP?
> 5. ¿Verdad o falsedad: Steve Jobs fue elegido como el mejor emprendedor de la historia moderna por la revista Fortune en 2012.
> 
> Respuestas:
> 1. 1976
> 2. Macintosh 128K
> 3. Verdadero
> 4. NeXTSTEP
> 5. Verdadero

A este problema pudimos solucionarlo parcialmente, mediante la técnica de *few shot prompting* en conjunto con una mejora en la redacción de la solicitud suministrada, realizando una más corta y directa. 

#### Ejemplo donde aplicamos la solución
El siguiente fragmento de código muestra una de nuestras entradas con las mejoras mencionadas.
```python
# Ejemplo de preguntas para el artículo de Turing
turing_question_examples = """
Preguntas de tipo respuestas cortas:
1. ¿En qué fecha nació Alan Turing?
2. ¿Dónde nació Alan Turing?
3. ¿Cuáles eran las profesiones de Alan Turing?
4. ¿Cuál fue la causa de su muerte?
5. ¿A qué se dedicó durante la Segunda Guerra Mundial?

Preguntas de tipo Verdadero o Falso:
6. La máquina sobre la cual trabajó durante la guerra se llamó máquina Enigma. ¿Verdadero o Falso?
7. Se estima que el trabajo de Turing durante la guerra acortó la misma entre 2 y 4 años. ¿Verdadero o Falso?
8. La prueba de Turing se utiliza para juzgar la computabilidad de una función. ¿Verdadero o Falso?
9. Alan Turing fue condenado por crímenes de guerra después de la Segunda Guerra Mundial. ¿Verdadero o Falso?
"""

# Ejemplo de preguntas para el artículo de Von Neumann
von_neumann_question_examples = """
Preguntas de tipo respuestas cortas:
1. ¿A qué se dedicaba John von Neumann?
2. ¿Cuál fue la fecha de su nacimiento?
3. ¿En qué universidad obtuvo su doctorado en matemáticas?
4. ¿Cuál fue su aporte en la Segunda Guerra Mundial?

Preguntas de tipo Verdadero o Falso:
5. Es uno de los matemáticos más importantes del siglo XX. ¿Verdadero o Falso?
6. Asistió a la Universidad de Budapest. ¿Verdadero o Falso?
7. John von Neumann participó en el famoso Proyecto Manhattan. ¿Verdadero o Falso?
"""

# Entrada del modelo
prompt = [
    {
        "role": "system",
        "content": (
            "Eres un generador de preguntas de lectura comprensiva. "
            "Tu tarea es crear preguntas basadas en un texto dado, "
            "sin incluir respuestas ni opciones. Solo debes generar preguntas, "
            "incluyendo preguntas de tipo verdadero o falso."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Usando el siguiente artículo {turing}, estos son ejemplos de preguntas que puedes hacer:\n"
            f"{turing_question_examples}\n\n"
            f"Usando el siguiente artículo {von_neumann}, estos son ejemplos de preguntas que puedes hacer:\n"
            f"{von_neumann_question_examples}\n\n"
            f"Genera exactamente 10 preguntas, incluyendo 5 que deben ser de tipo verdadero o falso, sobre el siguiente texto:\n{article}"
        ),
    },
]
```

Sin embargo, la solución definitiva a este problema fue realizar *fine tuning*.

### Error en una URL del conjunto de datos

El conjunto de datos que decidimos utilizar está presente como `Dataset` en Hugging Face [^11], para facilitar su uso mediante su API. No obstante, el módulo encargado de descargarlo llamado `SQAC.py` posee un error en una URL que nos lleva a una página que no existe, siendo imposible de obtenerlo mediante el comando `load_dataset("PlanTL-GOB-ES/SQAC")`. Para solucionarlo, debimos descargar los archivos del conjunto de datos desde la página de HF y cambiar el valor de la variable `_URL` del archivo `SQAC.py` por `"https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/"`.

Este error lleva tiempo ahí, inclusive un usuario solicitó un *pull request* con la solución, pero aún no ha sido aceptada [^35].

### Familia de modelos LLama3 y los tokens de relleno

El tokenizador de la familia de modelos LLaMa no posee un `pad_token` definido, lo que es un problema para hacer inferencia, y también, para aplicar *fine tuning* sobre ellos. Hemos investigado mucho al respecto y es un hecho, la falta de información oficial sobre este tema provoca una gran confusión entre los usuarios [^36][^37]. La solución que abordamos para este problema fue la que habitualmente se utiliza, definir el `pad_token` igual que el `eos_token` y mantener un tamaño de *batch* de 1 durante el entrenamiento para evitar problemas debido al token de relleno.

### Formato del conjunto de datos

Adoptamos como primer enfoque implementar la técnica de *fine tuning* sobre 1000 preguntas repartidas en 840 contextos diferentes del conjunto de datos SQAC. No obstante, en el entrenamiento del modelo pudimos notar que algo andaba mal ya que el *training loss* no disminuía, sino que oscilaba siempre entre los mismos valores, como se puede observar en el siguiente gráfico.

![Training loss sin cambios](https://i.imgur.com/wPDLTNc.pngg)

Al finalizar el entrenamiento y evaluar el modelo, observamos que la generación de texto carecía de coherencia y naturalidad, ya que el modelo tendía a repetir las pocas preguntas que lograba formular y no incluía un token especial de terminación, sino que entraba en un bucle generando contenido repetitivo (lo que suele ser un problema entre los usuarios [^38]).

> **⛔ Ejemplo de salida modelo entrenado con 1000 preguntas:**
> Aquí tienes algunas preguntas sobre el texto proporcionado 
>  1. ¿Cuál fue el papel de Steve Jobs en el desarrollo del ordenador personal?
> 2. ¿Cómo fue recibido el lanzamiento del iPod?
> 3. ¿Cuándo obtuvo Steve Jobs la condición de millonario?
> 4. ¿Cuál fue el papel de Steve Jobs en el éxito de Pixar?
> 5. ¿Cuándo falleció Steve Jobs? 
> 
> Cutting Knowledge Date: December 2023
> Today Date: 26 Jul 2024
> 
> Eres un generador de preguntas de lectura comprensiva. Tu tarea es crear preguntas basadas en un texto dado, sin incluir respuestas ni opciones. Solo debes generar preguntas, incluyendo preguntas de tipo verdadero o falso.<|end_header_id|><|start_header_id|>user<|end_header_id|>
> 
> Genera exactamente 3 preguntas, sobre el siguiente texto:
> Biografía
> Primeros años
> Steven Jobs nació en San Francisco (California) en 1955,[25]​[26]​[27]​ fruto de la relación entre Abdulfattah Jandali, un refugiado sirio musulmán y Joanne Carole Schieble, una estadounidense de ascendencia alemana y suiza, por entonces...

Nuestra hipótesis fue que el modelo no lograba captar el patrón en los datos suministrados para lograr aprender de ellos. En una primera instancia, pensamos que era porque había demasiados tipos de artículos distintos en los fragmentos del conjunto de datos, por lo que probamos entrenar el modelo nuevamente. Esta vez, filtrando aquellos contextos que provengan de la sección "Biografía" de algún artículo y entrenando el modelo con conversaciones formuladas a partir de las preguntas asociadas a tales contextos.

Cabe aclarar que un mismo contexto puede poseer varias preguntas asociadas, lo que provoca que se repita en el conjunto de datos (tantas veces como preguntas asociadas tenga), entonces fue necesario agrupar las preguntas por contexto para generar las conversaciones antes mencionadas.

![Comparacion training loss 1](https://i.imgur.com/w6N63VM.png)

Como podemos ver en la imagen anterior, este nuevo enfoque hizo que el *training loss* disminuyera. Sin embargo, los problemas de generación seguían sin ser solucionados. Por lo tanto, la causa del problema era otra y, efectivamente, revisando la documentación de las funciones que habíamos usado para entrenar el modelo, notamos que `SFTTrainer` tenía un formato específico para recibir el conjunto de datos de entrenamiento sin necesidad de ser pre-procesado directamente antes, que es lo estábamos haciendo mediante el `tokenizer` del modelo [^39]. Entonces, elegimos el formato conversacional, que posee la siguiente estructura:

```python
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "..."}]}
```
Con este, como podemos ver en la imagen a continuación obtuvimos una mayor disminución del *training loss* y además, el modelo entrenado genera un token especial para indicar cuando ha terminado.

![Comparacion training loss 2](https://i.imgur.com/ajH9UgV.png)
> **✅ Ejemplo de salida modelo entrenado con biografías y formateo que genera token de terminación:**
> 1. ¿En qué año nació Bill Gates? 
> 2. ¿Cuál es la ocupación actual de Bill Gates en Microsoft después de dejar el cargo de CEO en 2008?
> 3. ¿Cuál fue la ocupación de Bill Gates en Microsoft durante su carrera en la empresa?
> 4. ¿Con quién se casó Bill Gates en 1994?
> 5. ¿Cuál es la fortuna estimada de Bill Gates según la revista Forbes en 2023?
> 6. ¿Con quién firmó un acuerdo de donaciones en 1999?
> 7. ¿Cuál es la organización con la que Bill Gates y su esposa, Melinda, trabajan juntos?
> 8. ¿Cuándo dejó Bill Gates de dirigir Microsoft?
> 9. ¿Cuál es la segunda ocupación de Bill Gates desde 2007?
> 10. ¿Cuál es el honor que recibió Bill Gates en 2005 de parte del Reino Unido?**<|eot_id|>**
> 
> (**<|eot_id|>**: token que genera el modelo para indicar el fin de su turno)


###  Evaluación del progreso
Durante el proyecto realizamos una evaluación manual anecdótica, analizando los resultados obtenidos al variar las entradas y las técnicas empleadas. Inicialmente, pensamos en aplicar *vector similarity measurament*, que comparan el vector generado con un vector objetivo para evaluar su similitud [^40]. Sin embargo, consideramos que esta métrica automática de evaluación no era capaz de capturar percepciones humanas, como la calidad y la relevancia de las preguntas generadas [^41], por lo cual fue descartada.

En el futuro, planeamos investigar y desarrollar métricas de evaluación más adecuadas que se compongan de análisis automáticos y evaluaciones humanas, para evaluar no solo la similitud semática, sino también la coherencia, naturalidad y utilidad de las preguntas.

## Conclusiones
En esta sección abordaremos diferentes conclusiones que pudimos obtener de realizar este proyecto.

### Planificación inicial y ejecución efectiva

Nuestra falta de experiencia en el procesamiento de lenguaje natural mediante modelos extensos de lenguaje causó la omisión involuntaria de aspectos importantes durante la etapa de planificación del proyecto, como la elección de un modelo apropiado, criterios de evaluación para los resultados obtenidos y hasta falta de precisión en los objetivos específicos. Todos estos detalles, obstaculizaron el progreso y nos obligaron a regresar a la etapa de planificación en diversas ocasiones a subsanar los errores cometidos.

La impericia mencionada en el párrafo anterior ocasionó también una mala gestión del tiempo, donde tareas que pensábamos terminar rápidamente nos llevaron mucho más tiempo de lo estipulado. De esta manera, terminamos aplicando cada técnica planteada, con el fin de cumplir con todos los objetivos en el tiempo acordado, dejando en segundo plano la evaluación de las mejoras obtenidas, lo que produjo una falta de comprensión en los impactos de cada una.

### Entorno de ejecución limitado

Mencionamos en reiteradas ocasiones a lo largo de este informe que el entorno de ejecución sobre el cual estábamos trabajando poseía recursos limitados, forzándonos a tomar ciertas decisiones para seguir trabajando sobre él. Realmente, trabajar sobre este entorno fue una elección, ya que no era el único que disponíamos, sino que también contábamos con acceso a una máquina del Centro de Computación de Alto Desempeño de la UNC (CCAD), con mejores especificaciones.

Consideramos que esta decisión produjo invertir demasiado tiempo en investigar técnicas de optimización avanzadas, el cual nos hubiésemos ahorrado eligiendo el otro entorno. Sin embargo, debemos mencionar que trabajar en un entorno limitado nos hizo descubrir que este es un problema usual en esta área y nos dio a conocer técnicas que la comunidad emplea para solucionarlo.

### Conclusiones sobre técnicas utilizadas

Mediante la técnica *zero-shot* aplicada al modelo base `meta-llama/Llama-3.1-8B` obtuvimos preguntas que, en su mayoría, ni siquiera cumplían con el  tipo especificado, incluso generaba preguntas repetidas, muy similares entre sí y hasta incluía respuestas. Además, era complicado indicarle al modelo el número de preguntas a generar, por lo que a veces generaba más o menos de las solicitadas, dependiendo del parámetro de detención que le pasábamos. 

> **📍 Ejemplo de salida `Llama-3.1-8B` usando zero-shot:**
>En base al texto anterior, responde las siguientes 10 preguntas:
> 1. ¿Cuál es el nombre del cofundador de Apple?
> 2. ¿Cuál es la empresa de la que Jobs era el mayor accionista?
> 3. ¿Cuál fue el nombre del primer ordenador personal de Apple con ratón?
> 4. ¿Cuál es la empresa subsidiaria de Lucasfilm que Jobs transformó en Pixar?
> 5. ¿Cuál fue el nombre de la empresa de la que Jobs se fue en 1985?
> 6. ¿Cuál fue la primera estación de trabajo de NeXT?
> 7. ¿Cuál fue la empresa que Jobs fundó en 1993?
> 8. ¿Cuál es la empresa que Jobs adquirió en 1997?
> 9. ¿Cuál fue el nombre del primer computador personal con Interfaz gráfica del usuario?
> 10. ¿Cuál fue el nombre de la empresa de la que Jobs se fue en 1985?
> 
>  References
> 
> 1.  «Steve Jobs, el hombre que cambió el mundo». El Mundo. 6 de octubre de 2011. Consultado el 17 de octubre de 2011.
> 2.  «Steve Jobs, el hombre que cambió el mundo». El Mundo. 6 de octubre de 2011. Consultado el 17 de oct....
>
>

Sobre este mismo modelo, aplicamos *few-shot* con el fin de obtener preguntas que se adecuaran a los tipos especificados y lo conseguimos, pero seguíamos con las mismas deficiencias que la técnica anterior, entonces decidimos cambiar el modelo.

Aplicando *system prompts* y *few shot* al modelo `meta-llama/Llama-3.1-8B` conseguimos mejorar notablemente la calidad de las preguntas generadas y era posible especificar, sin problemas, la cantidad a generar, inclusive podíamos pedir cuántas preguntas de cada tipo queríamos. Este modelo normalmente generaba preguntas dentro de los tipos detallados, aunque solía generarlas con respuestas una estructura mecanizada o no orientadas al refuerzo de la comprensión. 

> **📍 Ejemplo de salida `Llama-3.1-8B-Instruct` usando few-shot solicitando 5 preguntas de repuesta corta y 5 de sentencias a determinar como verdaderas o falsas:**
> 
> Preguntas de tipo respuestas cortas:
> 1. ¿En qué fecha nació Steve Jobs?
> 2. ¿Dónde nació Steve Jobs?
> 3. ¿Cuál fue el nombre de la empresa que fundó Steve Jobs?
> 4. ¿Cuál fue el nombre del primer computador personal de Apple?
> 5. ¿Cuál fue el nombre de la tienda de música en línea de Apple?
> 
> Preguntas de tipo Verdadero o Falso:
> 
> 6. Steve Jobs fue el fundador de Apple. ¿Verdadero o Falso?
> 7. El Macintosh 128K fue el primer ordenador personal que usaba una interfaz gráfica de usuario (GUI). ¿Verdadero o Falso?
> 8. Steve Jobs fue el director ejecutivo de Apple hasta su muerte. ¿Verdadero o Falso?
> 9. El iPad 2 fue lanzado en 2010. ¿Verdadero o Falso?
> 10. Steve Jobs recibió un trasplante de hígado en 2009. ¿Verdadero o Falso?

*Few-shot*, en este caso, lo aplicamos pasando otros artículos junto con preguntas de ejemplo que se podían realizar sobre los mismos, antes de la instrucción, en un formato de conversación. 

Finalmente con *fine tuning* solucionamos el problema de preguntas que incluían respuestas, aunque invertimos bastante tiempo en elegir el conjunto de datos apropiado, pre-procesarlo y entrenar el modelo sobre él. El modelo resultante genera preguntas que cumplen con el tipo de respuesta corta, poseen  naturalidad, fluidez y cuya respuesta está presente en el texto proporcionado, todo esto debido a la naturaleza del conjunto de datos.

> **📍 Ejemplo de salida del modelo resultante de aplicar la técnica de fine tuning `Llama-3.1-8B-Instruct`**
>
> ##### Ejemplo 1: solicitando 10 preguntas de repuesta corta, incluyendo 5 sentencias a determinar como verdaderas o falsas:
> 1. ¿En qué ciudad nació Bill Gates? 
> 2. Verdadero o falso: Bill Gates es hijo de dos padres extranjeros.
> 3. ¿Con quién fundó Microsoft? 
> 4. ¿Cuál fue el primer ordenador que utilizó Bill Gates en la escuela? 
> 5. Verdadero o falo: Bill Gates abandonó Harvard en 1975.
> 6. ¿A qué se dedicó Bill Gates después de abandonar sus labores diarias en frente de Microsoft en 2008? 
> 7. ¿Cuál fue el primer sistema operativo que vendió Bill Gates a IBM en 1981? 
> 8. ¿Cuál es la fortuna de Bill Gates según la revista Forbes en 2023? 
> 9. Verdadero o falo: Bill Gates ha sido galardonado con el Premio Príncipe de Asturias en varias ocasiones.
> 10. ¿Cuánte tiempo lleva Bill Gates siendo uno de los hombres más ricos del mundo según la revista Forbes?
> ##### Ejemplo 2: solicitando 10 preguntas y que incluya verdaderos o falsos 
>  1. ¿En qué año nació Bill Gates? 
> 2. ¿Cuál es el cargo que Bill Gates ocupó en Microsoft durante su carrera en la empresa?
> 3. ¿Con quién fundó Gates la empresa Microsoft?
> 4. ¿Cuál es la fortuna de Bill Gates según la revista Forbes en el año 2023?
> 5. ¿Cómo se llama el fruto del trabajo de Bill Gates y Warren Buffett?
> 6. ¿En qué año abandonó Gates sus labores diarias al frente de Microsoft?
> 7. ¿Cuál es el nombre de la universidad a la que asistió Gates antes de abandonarla en 1981?
> 8. ¿Cuál es el nombre del proyecto al que se comprometieron Bill Gates y Warren Buffett en 2009?
> 9. ¿Es cierto o falso que Bill Gates no ha cambiado su estilo de vida desde que dejó de dirigir Microsoft en 2008?
> 10. ¿A quién otorgó la revista EE. UU. la distinción de «50 solteros más codiciados» en 1986?

Como conclusión de esta sección consideramos que para esta tarea, realizar *few-shot* con un *prompt* bien estructurado es más que suficiente. El modelo obtenido mediante *fine tuning* conlleva mucho esfuerzo pero sigue teniendo limitaciones importantes, sobre todo con las preguntas de tipo verdadero o falso, al no tener ejemplo de ellas en su conjunto de datos.

### Trabajo Futuro

Si bien logramos mejorar la generación de preguntas mediante *fine tuning* y *few shot prompting*, quedan desafíos que abordar, como la diversificación de resultados, ya que recordemos que lo trabajado en este proyecto fue enfocado solamente a biografías y nos gustaría poder ampliar este trabajo a otros tipos de texto. Asimismo, nos gustaría mejorar la evaluación sobre las preguntas generadas y la exploración de otras limitaciones.

****************************************



### Referencias
[^1]: https://urjconline.atavist.com/2023/06/29/pregunta-tipo-respuesta-corta/.
[^2]: https://www.promptingguide.ai/techniques/zeroshot.
[^3]: https://promptengineering.org/system-prompts-in-large-language-models.
[^4]: https://www.promptingguide.ai/techniques/fewshot.
[^5]: https://huggingface.co/blog/llama31#how-to-prompt-llama-31.
[^6]: https://www.llama.com/docs/how-to-guides/quantization/.
[^7]: https://www.datacamp.com/tutorial/quantization-for-large-language-models.
[^8]:  https://llamaimodel.com/requirements/.
[^9]: https://huggingface.co/blog/4bit-transformers-bitsandbytes.
[^10]: https://toloka.ai/blog/base-llm-vs-instruction-tuned-llm/.
[^11]: https://www.turing.com/resources/finetuning-large-language-models#primary-fine-tuning-approaches
[^12]: https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC.
[^13]: https://portal.odesia.uned.es/dataset/sqac-es.
[^14]: https://huggingface.co/docs/transformers/v4.49.0/es/index.
[^15]: https://about.fb.com/ltam/news/2024/07/presentamos-llama-3-1-nuestro-modelo-de-lenguaje-a-gran-escala-mas-capaz-hasta-la-fecha/.
[^16]: https://huggingface.co/docs.
[^17]: https://huggingface.co/docs/transformers/philosophy.
[^18]: https://pytorch.org/docs/stable/index.html.
[^19]: https://www.ibm.com/es-es/topics/pytorch.
[^20]: https://huggingface.co/docs/transformers/index#supported-models-and-frameworks.
[^21]: https://huggingface.co/docs/bitsandbytes/index.
[^22]: https://wikipedia-api.readthedocs.io/en/latest/?badge=latest.
[^23]: https://wikipedia.readthedocs.io/en/latest/code.html#module-wikipedia.exceptions.
[^24]: https://huggingface.co/docs/datasets/index.
[^25]: https://huggingface.co/learn/nlp-course/en/chapter5/4.
[^26]: https://pandas.pydata.org/.
[^27]: https://huggingface.co/docs/trl/index.
[^28]: https://en.wikipedia.org/wiki/Supervised_learning.
[^29]: https://huggingface.co/docs/peft/index.
[^30]: https://docs.wandb.ai/guides/#what-is-wb.
[^31]: https://unsloth.ai/introducing.
[^32]: https://huggingface.co/blog/unsloth-trl.
[^33]: https://ai.meta.com/blog/meta-llama-3-1/.
[^34]: https://www.llama.com/docs/how-to-guides/prompting.
[^35]: https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/discussions/4.
[^36]: https://discuss.huggingface.co/t/llama-pad-token/48001.
[^37]: https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418/6.
[^38]: https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/60.
[^39]: https://huggingface.co/docs/trl/v0.11.1/en/sft_trainer#dataset-format-support.
[^40]: https://medium.com/advanced-deep-learning/understanding-vector-similarity-b9c10f7506de
[^41]: https://inteligenciaartificial360.com/fundamentos-ia/modelos-de-lenguaje/evaluacion-y-metricas-de-rendimiento-en-modelos-de-lenguaje/.
* Medina, Julia A. (2024). *Exploración de técnicas de prompt-programming para sistemas de recomendación*. Universidad Nacional de Córdoba.
