import json, time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

for i in range(5):
    producer.send("ecom.events.raw", {"test": True, "i": i, "ts": int(time.time()*1000)})
producer.flush()
producer.close()
print("sent 5 test messages")
