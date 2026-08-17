package com.example.ocr_service.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

@Service
public class OcrResultConsumer {

    private static final Logger logger = LoggerFactory.getLogger(OcrResultConsumer.class);

    @KafkaListener(topics = "ocr_parsed_results", groupId = "ocr-backend-group")
    public void consumeParsedResults(String message) {
        logger.info("==========================================");
        logger.info("RECEIVED OCR PARSED RESULT");
        logger.info("==========================================");
        logger.info(message);
        logger.info("==========================================");
    }
}
