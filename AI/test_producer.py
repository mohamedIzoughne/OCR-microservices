import json
from confluent_kafka import Producer

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

if __name__ == '__main__':
    KAFKA_BROKER = 'localhost:9092'
    TOPIC = 'ocr_image_requests'

    producer_config = {
        'bootstrap.servers': KAFKA_BROKER
    }
    
    producer = Producer(producer_config)
    
    # Message to send
    payload = {
        "image_path": "images/store-4/image.png",
        "config_path": "configs/store-4.yaml"
    }
    
    print(f"Producing message to topic '{TOPIC}': {payload}")
    
    # Produce message
    producer.produce(
        TOPIC, 
        value=json.dumps(payload).encode('utf-8'), 
        callback=delivery_report
    )
    
    # Wait for any outstanding messages to be delivered and delivery report callbacks to be triggered.
    producer.flush()
