package com.example.ocr_service.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class KafkaProducerController {
    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    private static final String TOPIC = "ocr-topic";

    @GetMapping("/publish")
    public String publishMessage() {

        System.out.println("==========================================");
        System.out.println("KAFKA PRODUCER CONTROLLER");
        System.out.println("==========================================");

        kafkaTemplate.send(TOPIC,"Hello Kafka from spring boot")
            .whenComplete((result, exception) -> {
                if (exception == null) {
                    System.out.println("Message sent successfully to topic: " + result.getRecordMetadata().topic());
                } else {
                    System.err.println("Error sending message: " + exception.getMessage());
                }
            });
        return "Message published Successfully";
    }    
}
