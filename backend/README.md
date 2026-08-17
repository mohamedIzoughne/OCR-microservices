# Backend OCR Consumer Service

This backend service is built with Java and Spring Boot. It acts as an event-driven consumer that listens to the Kafka topic `ocr_parsed_results` and processes the results extracted by the Python AI pipeline.

## Prerequisites

- **Java 17** (or higher)
- **Maven** (optional, you can use the included wrapper `./mvnw`)
- A running **Kafka** broker (e.g. via `docker-compose up -d` at the root of the project).

## Configuration

The application properties are configured in `src/main/resources/application.properties`.
By default, the Kafka broker is expected at `localhost:9092`. You can override this using the environment variable `SPRING_KAFKA_BOOTSTRAP_SERVERS`.

## Running the Service

You can start the Spring Boot application using the Maven wrapper:

```bash
# On Linux/macOS
./mvnw spring-boot:run

# On Windows
mvnw.cmd spring-boot:run
```

Once running, the application will automatically connect to Kafka and print any OCR results produced by the AI service to the console.
