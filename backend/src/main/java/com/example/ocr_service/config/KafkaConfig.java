package com.example.ocr_service.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.core.ProducerFactory;

@Configuration
public class KafkaConfig {
    
    // This class enables sending messages to Kafka topics
    @Bean
    public KafkaTemplate<String, String> kafkaTemplate(ProducerFactory<String, String> producerFactory) {
        // Configure and return the KafkaTemplate bean
        return new KafkaTemplate<>(producerFactory);
    }   
}
