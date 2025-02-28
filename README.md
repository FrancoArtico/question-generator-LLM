# Generación de preguntas de lectura comprensiva en español utilizando LLM

#### Resumen
Este proyecto presenta una prueba de concepto para un generador de preguntas de lectura comprensiva en español, que se enfocará principalmente en dos tipos de preguntas: de respuesta corta [^1] y afirmaciones a evaluar como verdaderas o falsas. Para ello, utilizaremos el modelo de lenguaje `"Llama-3.1-8B-Instruct"` aplicando distintas técnicas y evaluando sus posibles mejorías en el desempeño. A lo largo de este proyecto trabajaremos principalmente sobre artículos de Wikipedia. Consideramos este estudio de interés dado a que la viabilidad de un sistema con estas características podría reducir la carga horaria de profesores y reforzar el aprendizaje de los estudiantes sobre los textos.

**Autores:** *Franco Artico* y *Nain Cadro*.

## Hipótesis, objetivos iniciales y el estado final alcanzado

Nuestra **hipótesis inicial** planteaba que el modelo seleccionado presentaría dificultades en la generación de preguntas de lectura comprensiva a partir de un texto en español, mostrando errores sintácticos o generando preguntas desalineadas a las estipuladas en el proyecto. Para poner a prueba nuestra hipótesis, primero investigamos y seleccionamos un modelo para evaluarlo. Además, fue imprescindible definir el tipo de preguntas que deseábamos generar y ciertos criterios para evaluarlas.

Con el modelo seleccionado (`meta-llama/Meta-Llama-3-8B-Instruct`) realizamos diversas pruebas las cuales mostraron que, efectivamente, presentaba deficiencias en la tarea solicitada las cuales pudieron ser identificadas y abordadas mediante las técnicas establecidas en los objetivos.

Durante el desarrollo del proyecto, ajustamos nuestro enfoque, reduciendo el dominio del problema a biografías o fragmentos de ellas. Este cambio nos permitió identificar dificultades más fácilmente en la generación de preguntas en comparación con trabajar con la extensa variedad de artículos disponibles en Wikipedia.

De esta manera, la siguiente lista es cómo quedaron constituidos los objetivos que nos propusimos alcanzar a lo largo del proyecto:
1. Investigar acerca de modelos de lenguaje entrenados para generar texto en español.
2. Seleccionar el tipo de texto específico sobre el cual el modelo realizará preguntas.
3. Definir críterios de evaluación para las preguntas generadas por el modelo de acuerdo con el objetivo del proyecto.
4. Detectar limitaciones del modelo elegido para la generación de preguntas en español.
5. Aplicar *zero-shot prompting* y evaluar los resultados obtenidos.
6. Aplicar *few shot prompting* y evaluar los resultados obtenidos.
7. Elegir un conjunto de datos apropiado para realizar el *fine tuning* del modelo.
8. Efectuar *fine tuning* sobre el modelo y buscar mejores resultados respecto al punto anterior.
9. Determinar las mejoras obtenidas en cada enfoque y realizar conclusiones acerca de las mismas.

A través de estos objetivos, logramos abordar las limitaciones detectadas del modelo y aplicar diferentes técnicas para mejorar la calidad de las preguntas generadas. Los resultados obtenidos mediante estas técnicas serán analizados con más detalle en las conclusiones del informe. 

## Técnicas relevantes

Existen diferentes técnicas para lograr que un modelo realice de forma correcta una tarea específica. En este proyecto se abordaron las siguientes técnicas: *zero-shot learning*, *few-shot learning*, *system prompts*, cuantización y *fine tuning*.

### Zero-shot learning
*Zero-shot learning* (sin ejemplos) es una técnica que consiste en interactuar con el modelo proporcionando la instrucción a realizar sin incluir ejemplos ni demostraciones de cómo queremos que sea realizada [^2].

Esta técnica la utilizamos inicialmente buscando descubrir la clase de preguntas generadas por los modelos base de la familia Llama 3.1, los cuales no poseen un entrenamiento específico en esta tarea.

Un modelo base no sigue instrucciones. Fue entrenado sobre una gran cantidad de datos para adquirir conocimientos básicos acerca del lenguaje, conocimiento general y contextos, por lo que es útil en situaciones donde debe dar respuestas de conocimiento general o conversaciones casuales. Sin embargo, para obtener lo que deseamos en este proyecto debemos solicitarlo de forma diferente, agregando indicaciones extras o guías adicionales [^3].

A continuación, presentamos un ejemplo de entrada que usamos para probar el modelo `meta-llama/Llama-3.1-8B`:

```python
prompt = article[0] + article[1] + "\n\n En base al texto anterior, responder las siguientes 5 preguntas:"
```

siendo `article[0]` y `article[1]` el primer y segundo párrafo respectivamente de un artículo de Wikipedia. De esta manera, incentivamos al modelo a generar preguntas respecto al texto que le proporcionamos.

### Few-shot learning

*Few-shot learining* es una técnica que consiste en pasarle al modelo, junto con la instrucción de la tarea, unos pocos ejemplos que sirvan como guía para aprender el patrón de los resultados que deseamos obtener [^4].

La idea de utilizar esta técnica en nuestro proyecto surge, principalmente, al darnos cuenta que los modelos naturalmente no generan preguntas de verdadero o falso. Entonces, necesitábamos proporcionarles ejemplos para incentivarlos a generarlas.

Un ejemplo de entrada que le proporcionamos al modelo usando este enfoque es el siguiente:

```python
prompt = (
    f"En base al siguiente texto:\n\n{article}\n\n"
    "Responde las siguientes 10 preguntas:\n"
    "1. ¿En qué fecha nació Alan Turing?\n"
    "2. ¿Dónde nació Alan Turing?\n"
    "3. La máquina sobre la cual trabajó durante la guerra se llamó máquina Enigma. ¿Verdadero o Falso?\n"
    "4. Se estima que el trabajo de Turing durante la guerra acortó la misma entre 2 y 4 años. ¿Verdadero o Falso?\n"
)
```

### System prompts

*System prompts* son un conjunto de instrucciones, guías e información acerca del contexto que se le provee al modelo antes de interactuar con las consultas del usuario [^3].

El uso de modelos base para la generación de preguntas presenta varias limitaciones. En primer lugar, no permite especificar de manera directa la cantidad y el tipo de preguntas a generar. Además, requieren estrategias más avanzadas de ingeniería de *prompts* para guiar de forma adecuada la generación de las preguntas, lo que implica un mayor esfuerzo en el diseño de instrucciones óptimas. Por estos motivos, se decidió optar por un modelo instruido como `meta-llama/Meta-Llama-3.1-8B-Instruct`, lo que también tendrá un impacto positivo en el *fine tuning*, ya que nos permite enfocarnos en mejorar el desempeño en la tarea específica, sin la necesidad de enseñarle al modelo a interpetar las instrucciones desde cero.

El cambio de modelo a `"meta-llama/Llama-3.1-8B-Instruct"` nos permitió aplicar esta técnica en nuestro proyecto, debido a que cuenta con ciertos roles, entre los cuales destacamos *system* y *user*. El rol *system* sirve para establecer el contexto de una conversación o proveerle al modelo información necesaria para ayudarle a responder efectivamente, mientras que el rol *user* representa la entrada del usuario [^5].

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
La cuantización es una técnica que nos permite comprimir modelos extensos de lenguaje representando sus pesos y activadores, los cuales están típicamente en números flotantes de 32-bits de precisión, con valores de menor precisión, usualmente 8-bits o 4-bits. La reducción en el número de bits provoca una gran disminución en el tamaño del modelo, por lo que consumen menos memoria, requieren menos espacio de guardado y son más rápidos de aplicarle *fine tuning* [^6][^7].

![imagen](https://i.imgur.com/3WSwSJF.png)

El modelo `"meta-llama/Llama-3.1-8B-Instruct"` requiere disponer de al menos 16 GB de RAM [^8]. Sin embargo, dado que nuestro entorno de ejecución en Google Colab solo ofrecía 15 GB de RAM, se hizo necesario aplicar cuantización para reducir el modelo. Para ello, implementamos la técnica NF4 (*4-bit Normal Float*), la cual cuantifica los pesos a 4 bits, permitiéndonos ejecutar y evaluar distintos modelos (como Llama2-7B, Llama3.1-8B y Llama3.1-8B-Instruct) pese a las limitaciones de recursos del entorno.

Además, un artículo de Hugging Face [^9], nos permitió saber antes de entrenar nuestro modelo que sería posible hacerlo en este entorno limitado usando GC (*gradient checkpointing*) y *nested quantization*.

### Fine tuning

*Fine tuning* es el proceso de ajustar los parámetros de un modelo extenso de lenguaje pre-entrenado para una tarea o dominio específico a través de un conjunto de datos. Esto dado que, los grandes modelos poseen un conocimiento amplio pero carecen de especialización en áreas específicas, por lo que esta técnica ataca ese problema haciendo a los modelos más acertados y efectivos para ciertos dominios o aplicaciones [^10][^11].

*Supervised Fine Tuning* es un método para mejorar y personalizar modelos pre-entrenados, la cual consiste en volver a entrenar estos modelos en un conjunto de datos etiquetado, compuesto por las instrucciones y las respuestas que se desean. Las tres técnicas más populares para aplicar SFT son *full fine-tuning*, *LoRA* y *QLoRA*.

![imagen](https://i.imgur.com/IG9s8Yk.png)

*QLoRA (Quantization-aware Low-Rank Adaptation)* es una técnica que en lugar de entrenar el modelo entero, congela los pesos e introduce pequeños adaptadores denominados *low-rank matrices* lo que le permite reducir mucho el número de parámetros a entrenar, reduciendo el uso de la memoria y el tiempo de entrenamiento respecto al *full fine-tuning*. Esta técnica es ideal para entornos donde la memoria GPU es limitada y por eso fue la elegida para entrenar nuestro modelo. La siguiente imagen ilustra la reducción de parámetros entrenables que esta técnica realizó en nuestro caso.

![](https://i.imgur.com/vqrHSB7.png)

El conjunto de datos que seleccionamos para esta técnica fue *Spanish Question Answering Corpus* (SQAC) [^12], que se compone de preguntas-respuestas extractivas en español. Sin embargo, lo consideramos ideal para la tarea de este proyecto, ya que posee contextos extraídos de artículos de Wikipedia, noticias de WikiNews y la sección española de AnCora corpus que es una mezcla de diferentes fuentes de noticias y literatura. Además, no posee preguntas sin respuestas y tanto las preguntas como las respuestas fueron anotadas por hablantes nativos de español con estudios universitarios [^13]. 

Ahora bien, a este conjunto de datos fue necesario realizarle un pre-procesamiento donde solamente nos quedamos con contextos provenientes de la sección "Biografías" de artículos de Wikipedia. Posterior a eso, nos quedamos con las columnas `context` y `question` para, a partir de ellas, armar una sola columna (`text`) representando una conversación entre un usuario que provee un contexto y solicita una cierta cantidad de preguntas, y un asistente que genera las preguntas a partir de ese contexto.

## Librerías exploradas y elecciones

### Transformers

![](https://i.imgur.com/jSFvf9J.png)

Transformers es una librería mantenida por Hugging Face y su comunidad, la cual proporciona APIs para descargar y entrenar modelos pre-entrenados. El uso de modelos pre-entrenados reduce los costos de cómputo y tiempo que conllevarían entrenar un modelo de cero [^14]. Seleccionamos esta librería tras elegir el modelo, la cual era mencionada en la página de Meta donde lo presentaban [^15]. 

La elección de esta librería se debió a diversos motivos, entre ellos podemos destacar el fácil acceso a modelos pre-entrenados, ya que mediante esta pudimos acceder a diversos modelos de forma simple, acelerando el desarrollo de la prueba de concepto. Además, la librería cuenta con mucha documentación [^16], tutoriales y una comunidad activa en sus foros, lo que facilitó el aprendizaje y la resolución de problemas que se presentaron durante el desarrollo. Otro motivo para su elección, fue que ofrece varias APIs para distintos niveles de conocimientos, lo que permitió adaptar el desarrollo a medida que aprendíamos. Asimismo, su sencillez nos permitió realizar inferencia, técnicas de *prompting* y ajustar modelos pre-entrenados en unas cuantas líneas de código, ahorrándonos tiempo y recursos [^17].

### PyTorch

PyTorch es una librería optimizada de tensores para aprendizaje profundo usando GPUs y CPUs [^18]. Los tensores, matemáticamente, son matrices multi-dimensionales y en este contexto, representan información aparentemente no numérica, como pueden ser textos o imágenes.

En el aprendizaje profundo se manejan grandes volúmenes de datos, esto sumado  a que íbamos a estar trabajando en un entorno con GPU, nos llevo a seleccionar PyTorch para manejar tensores dado a que permite realizar calculos de estos tanto en CPU como en GPU (contrario a los *ndarrays* de NumPy) [^19], y además es compatible con Transformers de Hugging Face para la familia de modelos Llama 3 [^20].

### Bitsandbytes

Los modelos extensos de lenguaje consumen muchos recursos para ser ejecutados y entrenados, por lo tanto el acceso a ellos es un desafío para los usuarios [^9]. Bitsandbytes es una librería que permite la cuantización en k-bits usando PyTorch, haciendo a los modelos extensos de lenguaje más accecibles [^21].

La elección de esta librería se debió principalmente a que permitía la cuantización en 4-bits usando NF4 y, tal como se menciona en su documentación, era ideal para aplicar la técnica de QLoRA en el *fine tuning* del modelo.

### Wikipedia-API

`Wikipedia-API` es un *wrapper* sencillo de usar para las API de Wikipedia que permite extraer textos, secciones, links, categorías, traducciones y demás [^22]. Habíamos explorado otro *wrapper* incluso más simple [^23], pero cuando necesitamos procesar los artículos antes de pasárselos al modelo para hacer inferencia, comenzamos a tener dificultades para eliminar secciones no relevantes correctamente.

Usamos el *wrapper* seleccionado para obtener los artículos sobre los cuales luego le solicitábamos al modelo realizar las preguntas.

### Datasets y Pandas
Datasets es una librería para acceder y compartir fácilmente conjuntos de datos para diversas tareas, entre ellas el procesamiento de lenguaje natural [^24].

Elegimos utilizar esta librería debido a que nos proveía acceso al conjunto de datos que elegimos, es fácil de utilizar y provee métodos los cuales son poderosos para el procesamiento de datos (como `map` y `filter`). Además, está diseñada para manejar de forma eficiente grandes conjuntos de datos, evitando cargarlos directamente en la memoria RAM usando Apache Arrow [^25].

Sin embargo, no fue posible sacarle provecho al máximo a esta elección. En un momento tomamos la decisión de convertir nuestro conjunto de datos en un DataFrame de Pandas [^26], el motivo fue que nos facilitaba agrupar las preguntas por contextos mediante el método`groupby`. No obstante, esto no representó ninguna dificultad, ya que ambas librerías cuentan con métodos para convertir entre DataFrames y Datasets.

### TRL

TRL provee herramientas para entrenar modelos de lenguaje que posean la arquiectura de Tranformers mediante aprendizaje de refuerzo [^27].

La elección de esta libreria fue crucial para aplicar *supervised fine tuning* [^28], la cual es una técnica que consiste en adaptar un modelo extenso de lenguaje pre-entrenado a una tarea específica usando datos etiquetados. En nuestro caso, los datos etiquetados estaban conformados por preguntas del conjunto de datos SQAC y TRL mediante la clase `SFTTrainer` nos facilitó poner esta técnica en práctica en nuestro proyecto.

Otros motivos de la elección de esta libreria fueron su integración con la librería Transformers de Hugging Face, la amplia documentación que posee, el soporte con PEFT, libreria cuya importancia detallaremos en la próxima sección.

### PEFT
PEFT es una librería que nos permite adaptar de manera eficiente modelos extensos de lenguaje sin entrenar todo los párametros del modelo (*full fine-tuning*). En su lugar, los métodos de PEFT solo entrenan una pequeña cantidad de parámetros extra, reduciendo los costos de entrenamiento en términos recursos computacionales y de almacenamiento, con un rendimiento comparable a haber entrenado el modelo completo [^29].

Optamos por esta librería debido a su integración con Transformers  las herramientas que proporciona para configurar y aplicar QLoRA de manera sencilla.

### Wandb

Es una librería de la plataforma *Weights & Biases* la cual ofrece herramientas para entrenar modelos, afinarlos y aprovechar los modelos fundamentales [^30].

Esta plataforma nos permitó llevar un seguimiento de los hiper-parámetros  utilizandos para entrenar el modelo junto con un monitoreo en tiempo real del comportamiento del mismo durante el entrenamiento. Asimismo, ofreció un registro de varias métricas del modelo como el *training loss* mediante gráficos muy útiles. Todo esto, además facilitó la comparación de diferentes entrenamientos realizados.

### Unsloth

Unsloth es una herramienta que promete entrenar modelos extensos de lenguaje más rápido y con menos recursos, sin pérdida de *accuracy* [^31]. En nuestro caso, intentamos usarlo para inferencia primero. Como resultado, obtuvimos una velocidad de inferencia mayor comparada a la realizada sin el uso de esta herramienta, pero notamos una degradación en la calidad de respuesta usando contextos largos (alrededor de 15.000 tokens), generando resúmenes en lugar de preguntas sobre del texto solicitado, por lo que optamos por no seguirla utilizando.

Nuestra hipótesis sobre esta degradación se basa en la forma en la que Unsloth maneja los contextos a través de *RoPE Scaling* [^32]. Consideramos que, al procesar contextos de este tamaño, el modelo podría perder la capacidad de retener la instrucción solicitada, tendiendo en su lugar a realizar la instrucción más común asociada con dicha longitud de contexto.

## Bases de código exploradas

Dado a que era nuestra primera vez trabajando en un proyecto sobre el procesamiento de lenguaje natural, fue necesario explorar diferentes bases de código para aprender las técnicas más utilizadas, las diferentes partes en el desarrollo y para mantener nuestro código simple y fácil de leer pero modernizado, ya que varias de las librerias utilizadas en el proyecto se actualizan de forma regular.



| Nombre                                                                                  | Repositorio                                                                                                      | Propósito                                                                          | Aspectos clave                                                                              | Motivo de análisis                                                                                                                              |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
|    🤗 **Tokenizers - Hugging Face**                                                                                       |                                                                                             [huggingface/tokenizers](https://github.com/huggingface/tokenizers)                     | Pre-procesameinto de datos, inferencia y entrenamiento.                                                                                    |     Buena documentación y muchos ejemplos.                                                                                        | Ofrece una implementación optimizada para la tokenización rápida y eficiente, compatible con los modelos del Hub. |
| 🤗 **Datasets - Hugging Face**                                                          | [huggingface/datasets](https://github.com/huggingface/datasets)                                                  | Descarga y preprocesamiento de datos.                                              | Carga de conjunto de datos públicos de forma sencilla y pre-procesado eficiente.            | Ofrece código para descargar conjuntos de datos y diferentes configuraciones para hacerlo.                                                      |
| 🤗 **TRL - Hugging Face**                                                               | [huggingface/trl](https://github.com/huggingface/trl)                                                            | *Fine tuning* usando aprendizaje supervisado.                                      | Soporte con PEFT, buena documentación, eficiencia y `Trainers`.                             | Brinda implementaciones de entrenamientos, ejemplos de formatos de conjunto de datos utilizados para entrenar y de configuración de parámetros. |
| 🤗 **NLP Course - Hugging Face**                                                        | [huggingface/course](https://github.com/huggingface/course)                                                      | Inferencia de los distintos modelos probados y comprensión de conceptos claves.    | Aprendizaje guiado sobre el ecosistema de HuggingFace y conceptos clave de NLP.             | Instruirnos en el uso de `Transformers`, `Dataset` y `Tokenizers`.                                                                              |
| 🤗 **Transformers - Hugging Face**                                                      | [huggingface/transformers](https://github.com/huggingface/transformers)                                          | Implementación de modelos LLM.                                                     | Compatibilidad con modelos Llama y comunidad activa.                                        | Base para hacer inferencia probando modelos mediante `pipeline`.                                                                                |
| 📖 **LLM Course**                                                                       | [mlabonne/llm-course](https://github.com/mlabonne/llm-course)                                                    | *Fine tuning* y pre-procesamiento de datos.                                        | Explicaciones detalladas, entrenamiento ultra-eficiente y código fácil de leer.             | Aprender cómo entrenar el modelo de manera eficiente, usando técnicas de aprendizaje supervisado.                                               |
| 🤗 **PEFT - Hugging Face**                                                              | [huggingface/peft](https://github.com/huggingface/peft)                                                          | *Fine-tuning* usando QLoRA y valor de parámetros para la aplicación de la técnica. | *Fine-tuning* eficiente y buena documentación.                                              | Ofrece explicación detallada acerca de los parámetros para aplicar QLoRA y código donde se aplica la técnica.                                   |
| ⚡ **Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA** | [colab/bnb-4bit-training](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) | *Fine tuning* de modelos mediante QLoRA y cuantización en NF4.                     | Uso del ecosistema de HuggingFace, *nested quantization*, NF4 y *Gradient checkpointing*.   | Ofrece una implementación optimizada usando QLoRA y es compatible con HuggingFace.                                                              |
| 🦥 **Unsloth**                                                                          | [unslothai/unsloth](https://github.com/unslothai/unsloth)                                                        | Optimización para realizar inferencia y *fine tuning*.                             | Optimización de la memoria, compatibilidad con modelos Llama 3.1 y velocidad de inferencia. | Brinda código para entrenar y hacer inferencia de modelos de la familia Llama 3, usando la herramienta Unsloth en un entorno de HuggingFace.    |
| 📚 **Wikipedia-API**                                                                    | [project/wikipedia-API](https://pypi.org/project/Wikipedia-API/)                                                 | Obtención de datos de Wikipedia para probar modelos.                               | Simpleza y facilidad para recuperar información de Wikipedia.                               | Ofrece código sobre el uso del *wrapper* mediante casos de uso.                                                                                 |

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

Debíamos buscar modelos que tengan soporte para la generación de texto en español. En dicha búsqueda, encontramos a la familia de modelos LLaMa en su versión 3.1 que son multilingües [^33]. Para comprender la importancia de este punto, en la siguiente subsección mostraremos una comparativa sencilla entre la capacidad de generación de texto en español del modelo `meta-llama/Llama-2-13b-hf`, que no tiene soporte oficial para el idioma, y `meta-llama/Llama-3.1-8B`.

Además, como mencionamos en la sección *system prompt*, terminamos eligiendo el modelo `meta-llama/Llama-3.1-8B-Instruct` ya que nos otorgaba más facilidades para solicitarle instrucciones al modelo.

#### Comparativa: soporte para la generación de texto en español

La comparativa constó en solicitarle a ambos modelos completar una sentencia que comenzaba con `"El perro es"`, con los mismos parámetros de generación, entre los cuales incluimos un número para `max_new_tokens` para así evitar secuencias generadas que sean muy largas. La siguiente imagen es un ejemplo de los textos generados por ambos modelos.

![Compartiva](https://i.imgur.com/UsBXOZ6.png)

Como ya hemos mencionado a lo largo del artículo, la elección del modelo  `"meta-llama/Llama-3.1-8B-Instruct"` fue clave para el *prompting*, debido a que puede interpretar instrucciones y usar roles en su entrada para adaptar su salida [^34].

### Reducir el dominio

Los artículos presentes en Wikipedia pueden tratarse de diversos temas, asimismo una pregunta puede ser menos relevante en un artículo que en otro. Es por eso que decidimos reducir el dominio de los artículos a solo biografías, sobre los cuales evaluaremos las preguntas generadas. De esta manera, sería más fácil evaluar la calidad de las preguntas y detectar las limitaciones del modelo.

### Respuestas no solicitadas junto con las preguntas generadas

La tarea que le solicitamos al modelo siempre fue generar preguntas a partir de un determinado contexto. Uno de los problemas que surgieron con esta solicitud fue que el modelo también generaba las respuestas de dichas preguntas, inclusive si le especificábamos explícitamente que no lo haga, a través del rol `system`. A este problema pudimos solucionarlo parcialmente, mediante la técnica de *few shot prompting* en conjunto con una mejora en la redacción de la solicitud suministrada realizando una más corta y directa. Sin embargo, la solución definitiva a este problema fue el *fine tuning*.

### Error en una URL del conjunto de datos

El conjunto de datos que decidimos utilizar está presente como `Dataset` en Hugging Face [^11], para facilitar su uso mediante su API. No obstante, el módulo encargado de descargarlo llamado `SQAC.py` posee un error en una URL que nos lleva a una página que no existe y lo cual no permite obtenerlo mediante el comando `load_dataset("PlanTL-GOB-ES/SQAC")`. Para solucionarlo, debimos descargar los archivos del conjunto de datos desde la página de HF y cambiar el valor de la variable `_URL` del archivo `SQAC.py` por `"https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/"`.

Este error lleva tiempo ahí, inclusive un usuario solicitó un *pull request* con la solución, sino que aún no ha sido aceptada [^35].

### Familia de modelos LLama3 y los tokens de relleno

El tokenizador de la familia de modelos LLaMa, no poseen un `pad_token` definido, lo que es un problema para hacer inferencia, y también, para aplicar *fine tuning* sobre ellos. Hemos investigado mucho al respecto, y es un hecho, la falta de información oficial sobre este tema provoca una gran confusión entre los usuarios [^36][^37]. La solución que abordamos para este problema fue la que habitualmente se utiliza, definir el `pad_token` igual que el `eos_token` y mantener un tamaño de *batch* de 1 durante el entrenamiento para evitar problemas debido al token de relleno.

### Formato del conjunto de datos

Adoptamos como primer enfoque implementar la técnica de *fine tuning* sobre 1000 preguntas repartidas en 840 contextos diferentes del conjunto de datos SQAC. No obstante, en el entrenamiento del modelo pudimos notar que algo andaba mal debido a que el *training loss* no disminuía, sino que oscilaba siempre entre los mismos valores, como se puede observar en el siguiente gráfico:

![Training loss sin cambios](https://i.imgur.com/wPDLTNc.pngg)

Al finalizar el entrenamiento y evaluar el modelo, observamos que la generación de texto carecía de coherencia y naturalidad. Además, el modelo tendía a repetir las pocas preguntas que lograba formular y no incluía un token especial de terminación, sino que entraba en un bucle generando contenido repetitivo (lo que suele ser un problema entre los usuarios [^38]).

Nuestra hipótesis fue que el modelo no lograba captar el patrón en los datos suministrados para lograr aprender de ellos. En una primera instancia, pensamos que era porque había demasiados tipos de artículos distintos en los fragmentos del conjunto de datos, por lo que probamos entrenar el modelo nuevamente, pero esta vez, filtrando aquellos contextos que provengan de la sección "Biografía" de algún artículo, y entrenando el modelo con conversaciones formuladas a partir de las preguntas asociadas a tales contextos.

Cabe aclarar que un mismo contexto puede poseer varias preguntas asociadas, lo que provoca que se repita varias veces en el conjunto de datos (tantas veces como preguntas asociadas tenga), entonces fue necesario agrupar las preguntas por contexto para generar las conversaciones antes mencionadas.

![Comparacion training loss 1](https://i.imgur.com/w6N63VM.png)

Como podemos ver en la imagen anterior, este nuevo enfoque hizo que el *training loss* disminuyera. Sin embargo, los problemas de generación seguían sin ser solucionados. Por lo tanto, la causa del problema era otra y efectivamente, revisando la documentación de las funciones que habíamos usado para entrenar el modelo notamos que `SFTTrainer` tenía un formato específico para recibir el conjunto de datos de entrenamiento sin necesidad de ser pre-procesado directamente antes, que es lo estábamos haciendo mediante el `tokenizer` del modelo [^39]. Entonces, elegimos el formato conversacional el cual tiene la siguiente estructura:

```python
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "..."}]}
```
Con el cual, como podemos ver en la imagen a continuación obtuvimos una mayor disminución del *training loss* y además, el modelo entrenado genera un token especial para indicar cuando ha terminado.

![Comparacion training loss 2](https://i.imgur.com/ajH9UgV.png)


###  Evaluación del progreso
Hasta ahora se realizó una evaluación anecdótica manual comparando los diferentes resultados obtenidos al cambiar las entradas y las técnicas utilizadas. Se planea para un futuro investigar alternativas de evaluación.

## Conclusiones
En esta sección abordaremos diferentes conclusiones que pudimos obtener de realizar este proyecto.

### Planificación inicial y ejecución efectiva

Nuestra falta de experiencia en el procesamiento de lenguaje natural mediante modelos extensos de lenguaje causó la omisión involuntaria de aspectos importantes durante la etapa de planificación del proyecto, como la elección de un modelo apropiado, críterios de evaluación de los resultados obtenidos y hasta falta de precisión en los objetivos específicos. Todos estos detalles, obstaculizaron el progreso y nos obligaron a regresar a la etapa de planificación en diversas ocaciones a subsanar los errores cometidos.

La impericia mencionada en el párrafo anterior ocacionó también una mala gestión del tiempo, donde tareas que pensábamos terminar rápidamente nos llevaron mucho más tiempo de lo estipulado. De esta manera, terminamos aplicando cada técnica planteada, con el fin de cumplir con todos los objetivos en el tiempo acordado, dejando en segundo plano la evaluación de las mejoras obtenidas, lo que produjo una falta de comprensión en los impactos de cada una.

### Entorno de ejecución limitado

Mencionamos en reiteradas ocaciones a lo largo de este informe que el entorno de ejecución sobre el cual estábamos trabajando poseía recursos limitados, forzándonos a tomar ciertas decisiones para seguir trabajando sobre él. Realmente, trabajar sobre este entorno fue una decisión, ya que no era el único que disponiamos, sino que también contábamos con acceso a una máquina del Centro de Computación de Alto Desempeño de la UNC (CCAD), con mejores especificaciones.

Consideramos que esta decisión ocacionó que debamos invertir demasiado tiempo en investigar técnicas de optimización avanzadas, el cual  nos hubiésemos ahorrado eligiendo el otro entorno. Sin embargo, debemos mencionar que trabajar en un entorno limitado nos hizo descrubrir que este es un problema usual en esta área y nos dió a conocer técnicas que la comunidad emplea para solucionarlo.

### Conclusiones sobre técnicas utilizadas

Mediante la técnica *zero-shot* aplicada al modelo base `meta-llama/Llama-3.1-8B` obtuvimos preguntas las cuales, en su mayoria, ni siquiera cumplían con el  tipo especificado, incluso generaba preguntas repetidas, muy similares entre sí y hasta incluía respuestas. Además, era complicado indicarle al modelo el número de preguntas a generar, por lo que a veces generaba más o menos, dependiendo del parámetro de detención que le pasábamos. Sobre este mismo modelo, aplicamos *few-shot* con el fin de obtener preguntas que se adecuaran a los tipos especificados y lo conseguimos, pero seguíamos con las mismas deficiencias que la técnica anterior, entonces decidimos cambiar el modelo.

Aplicando *system prompts* y *few shot* al modelo `meta-llama/Llama-3.1-8B` conseguímos mejorar notablemente la calidad de las preguntas generadas y era posible especificar, sin problemas, la cantidad de preguntas a generar, es más, podíamos pedir cuántas preguntas de cada tipo queríamos. El modelo, normalmente, generaba preguntas dentro de los tipos detallados, aunque, solía generar preguntas que tenían respuestas abiertas, no orientadas al refuerzo de la comprenión o incluyendo las respuestas. *Few-shot*, en este caso, lo aplicamos pasando otros artículos junto con preguntas de ejemplo que se podían realizar sobre los mismos antes de la instrucción en un formato de conversación. 

Finalmente con *fine tuning* solucionamos el problema de preguntas que incluían respuestas, aunque invertimos bastante tiempo en elegir el conjunto de datos apropiado, pre-procesarlo y entrenar el modelo sobre él. El modelo resultante genera preguntas que cumplen con el tipo de respuesta corta, poseen  naturalidad, fluidez y cuya respuesta está presente en el texto proporcionado, todo esto debido a la naturaleza del conjunto de datos.

Como conclusión de esta sección consideramos que para esta tarea, realizar *few-shot* con un *prompt* bien estructurado es más que suficiente. El modelo obtenido mediante *fine tuning* conlleva mucho esfuerzo pero sigue teniendo limitaciones importantes, por ejemplo al no tener preguntas de verdadero o falso en su conjunto de datos tiende a no generarlas.

### Trabajo Futuro

Si bien logramos mejorar la generación de preguntas mediante *fine tuning* y *few shot prompting*, quedan desafíos que abordar como la diversificación de resultados, ya que recordemos que lo trabajado en este proyecto fue enfocado solamente a biografías, nos gustaría poder ampliar este trabajo a otros tipos de texto. Asimismo, nos gustaría mejorar la evaluación las preguntas generadas y la exploración de otras limitaciones.


****************************************

### Bibliografía

Medina, Julia A. (2024). *Exploración de técnicas de
prompt-programming para sistemas
de recomendación*. Universidad Nacional de Córdoba.

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