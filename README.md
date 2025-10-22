# foodDeliveryDispatch
1.	Overview
This simulation models a food delivery platform (similar to UberEats, PickMe, DoorDash, or Deliveroo) to analyze dispatch strategies and fleet management performance.

2.	System Description

Components
1.	Orders: Customer food orders with restaurant and delivery locations
2.	Drivers: Fleet of delivery personnel who pick up and deliver orders
3.	Dispatch System: Assigns orders to drivers using different strategies
4.	Service Area: 10km × 10km geographic region

Key Features
•	Realistic order generation: Poisson arrival process
•	Multiple dispatch strategies: Nearest, FCFS, Balanced
•	Geographic simulation: 2D coordinates for restaurants, customers, drivers
•	Time-based metrics: Wait times, delivery times, driver utilization
•	Workload balancing: Fair distribution across drivers
