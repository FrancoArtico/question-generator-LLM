# Generación de preguntas de lectura comprensiva en español utilizando LLM

*Franco Artico y Nain Cadro*


*Universidad Nacional de Córdoba*

La generación de preguntas es una tarea de procesamiento de lenguaje natural cuyo objetivo es producir preguntas válidas de acuerdo con un pasaje de texto dado y la respuesta a la cual aspiramos [1]. La utilidad de este tipo de tarea se ve reflejada en el ámbito educativo, dado que, en caso de aplicarse, los docentes podrían reducir la carga de trabajo y enfocar sus horas en otras tareas de aprendizaje. Mientras que los estudiantes pueden usarlo para autoevaluación o bien, para poner a prueba sus conocimientos en materiales de estudio que no tengan una guía práctica.

Para llevar a cabo el proyecto, decidimos usar un Modelo Extenso de Lenguaje (LLM), debido a que se encuentran pre entrenados con grandes cantidades de datos, lo que le aporta contexto suficiente para generar mejores preguntas comparado con otros enfoques utilizados previamente [2][3]. A este modelo le aplicaremos *prompt engineering* y *fine tuning* con el objetivo de mejorar las preguntas obtenidas y comparar sus resultados.

### Hipótesis del trabajo

Este proyecto supone que la tarea de generación de preguntas en base a una LLM posee ciertas limitaciones, las cuales podrán ser identificadas y abordadas por distintas técnicas para ser reducidas. Esto permitirá generar preguntas de mayor relevancia y calidad en base al texto proporcionado, disminuyendo los errores.  
En resumen, partiremos de un LLM sobre el cual aplicaremos *few shot prompting* y por otro lado, *fine tuning;* luego analizaremos sus resultados y llegaremos a la conclusión de que este último mejora respecto al anterior.  
Además, las preguntas generadas tendrán una respuesta, que será la proporcionada como entrada del modelo junto con el contexto de la misma.

### Objetivos preliminares

* Detectar limitaciones de LLaMa2 para la generación de preguntas en español.  
* Realizar técnica de *few shot* *prompting* y evaluar los resultados obtenidos.  
* Efectuar *fine tuning* sobre LLaMa2 y buscar mejores resultados respecto al punto anterior.  
* Determinar las mejoras obtenidas en cada enfoque y realizar conclusiones acerca de las mismas.

### Técnicas a utilizar

*Few shot prompting* es una técnica que consiste en dar al modelo varios ejemplos para mejorar su capacidad de respuesta en tareas específicas.  
*Fine tuning* es el proceso de adaptar un modelo previamente entrenado para tareas o casos de uso específicos.  
*spaCy vector similarity* para medir la similtud entre las respuestas deseadas y las generadas por el modelo.

### Evaluación

La evaluación será anecdótica manual y tendremos en cuenta el utilizar un conjunto de datos para realizar una comparación de similitud de las respuestas obtenidas por el modelo, con las presentes en el conjunto de datos, utilizando *spaCy vector similarity*.

### Planificación semanal

#### Investigación y preparación (semana 1\)

* Determinar la versión del modelo del lenguaje a utilizar.  
* Buscar y leer paper sobre generación de preguntas.  
* Leer y aprender sobre las técnicas de *fine tuning* y *few shot prompting*.
* Determinar criterios de evaluación y métricas.  
* Buscar conjunto de datos de preguntas en español.  
* Instalar y configurar el modelo determinado junto con las librerías.

#### Estudiar el modelo (semana 2\)

* Hacer una evaluación empírica de las preguntas generadas y evaluar las limitaciones del modelo.  
* Documentar limitaciones y proponer soluciones para superarlas.  
* Curación del conjunto de datos.

#### Few shot prompting (semana 3\)

* Desarrollar prompts buscando mejorar la generación de las preguntas.  
* Determinar aquellos que generen mejores resultados.  
* Documentar el proceso y los resultados.

#### Fine tuning (semana 4\)

* Realizar *fine tuning* del modelo con el conjunto de datos ya curados.  
* Evaluar las preguntas generadas y comparar con los resultados previos logrados con few shot prompting y mediante el modelo.  
* Documentar el proceso, observaciones y resultados.

#### Conclusiones (semana 5\)

* Elaborar una conclusión donde se comparen los resultados y el costo de aplicar  cada una de las técnicas.  
* Hacer correcciones finales sobre las netbooks.

### Referencias

1. [Papers With Code: Question Generation](https://paperswithcode.com/task/question-generation).
2. [Learning to Ask: Naural Question Generation for Reading Comprehension (Arxiv)](https://arxiv.org/pdf/1705.00106).
3. [Prompt-Engineering and Transformer-based Question Generation and Evaluation (Arxiv)](https://arxiv.org/pdf/2310.18867).
4. [Meta AI: 5 Steps to Getting Started with Llama 2](https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/).  
5. [huggingfaceLlama2](https://huggingface.co/docs/transformers/main/model_doc/llama2).
