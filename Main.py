"""
Food Delivery Dispatch System Simulation
A case study on delivery fleet management and dispatch strategies
Author: Case Study Project
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import seaborn as sns

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DriverStatus(Enum):
    """Driver availability states"""
    AVAILABLE = "available"
    PICKING_UP = "picking_up"
    DELIVERING = "delivering"


class DispatchStrategy(Enum):
    """Dispatch assignment strategies"""
    NEAREST = "nearest"  # Assign to nearest available driver
    FCFS = "fcfs"  # First-Come-First-Served (first available driver)
    BALANCED = "balanced"  # Balance workload across drivers


@dataclass
class Location:
    """Represents a 2D location"""
    x: float
    y: float

    def distance_to(self, other: 'Location') -> float:
        """Calculate Euclidean distance to another location"""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Order:
    """Represents a food delivery order"""
    order_id: int
    arrival_time: float
    restaurant_location: Location
    customer_location: Location
    prep_time: float  # Food preparation time at restaurant
    assigned_time: Optional[float] = None
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    driver_id: Optional[int] = None

    @property
    def wait_time(self) -> Optional[float]:
        """Time from order to assignment"""
        if self.assigned_time:
            return self.assigned_time - self.arrival_time
        return None

    @property
    def total_time(self) -> Optional[float]:
        """Time from order to delivery"""
        if self.delivery_time:
            return self.delivery_time - self.arrival_time
        return None

    @property
    def delivery_distance(self) -> float:
        """Total distance: driver->restaurant + restaurant->customer"""
        return self.restaurant_location.distance_to(self.customer_location)


@dataclass
class Driver:
    """Represents a delivery driver"""
    driver_id: int
    current_location: Location
    status: DriverStatus = DriverStatus.AVAILABLE
    current_order: Optional[Order] = None
    completed_orders: List[Order] = None
    total_distance: float = 0.0
    available_at: float = 0.0

    def __post_init__(self):
        if self.completed_orders is None:
            self.completed_orders = []


class DeliverySystem:
    """
    Simulates a food delivery dispatch system
    """

    def __init__(self, num_drivers=10, dispatch_strategy=DispatchStrategy.NEAREST,
                 service_area_size=10.0, avg_speed=0.5):
        """
        Initialize the delivery system

        Args:
            num_drivers: Number of delivery drivers
            dispatch_strategy: Strategy for assigning orders to drivers
            service_area_size: Size of square service area (km)
            avg_speed: Average driver speed (km/minute)
        """
        self.num_drivers = num_drivers
        self.dispatch_strategy = dispatch_strategy
        self.service_area_size = service_area_size
        self.avg_speed = avg_speed  # km per minute

        # Initialize drivers at random locations
        self.drivers = []
        for i in range(num_drivers):
            location = Location(
                x=np.random.uniform(0, service_area_size),
                y=np.random.uniform(0, service_area_size)
            )
            self.drivers.append(Driver(driver_id=i, current_location=location))

        # System state
        self.current_time = 0.0
        self.pending_orders = []
        self.completed_orders = []
        self.total_orders_received = 0

        # Performance tracking
        self.wait_times = []
        self.delivery_times = []
        self.driver_utilization = {i: [] for i in range(num_drivers)}

    def add_order(self, order: Order):
        """Add new order to the system"""
        self.pending_orders.append(order)
        self.total_orders_received += 1

    def find_available_drivers(self) -> List[Driver]:
        """Get list of currently available drivers"""
        return [d for d in self.drivers if d.status == DriverStatus.AVAILABLE
                and d.available_at <= self.current_time]

    def assign_order_nearest(self, order: Order, available_drivers: List[Driver]) -> Optional[Driver]:
        """Assign order to nearest available driver"""
        if not available_drivers:
            return None

        # Find nearest driver to restaurant
        nearest_driver = min(available_drivers, key=lambda d: d.current_location.distance_to(order.restaurant_location))
        return nearest_driver

    def assign_order_fcfs(self, order: Order, available_drivers: List[Driver]) -> Optional[Driver]:
        """Assign order to first available driver (FCFS)"""
        if not available_drivers:
            return None
        return available_drivers[0]  # First in list

    def assign_order_balanced(self, order: Order, available_drivers: List[Driver]) -> Optional[Driver]:
        """Assign order to driver with least completed orders"""
        if not available_drivers:
            return None

        # Balance workload
        return min(available_drivers, key=lambda d: len(d.completed_orders))

    def assign_orders(self):
        """Attempt to assign pending orders to available drivers"""
        available_drivers = self.find_available_drivers()

        orders_to_remove = []

        for order in self.pending_orders:
            if not available_drivers:
                break

            # Select driver based on strategy
            if self.dispatch_strategy == DispatchStrategy.NEAREST:
                driver = self.assign_order_nearest(order, available_drivers)
            elif self.dispatch_strategy == DispatchStrategy.FCFS:
                driver = self.assign_order_fcfs(order, available_drivers)
            elif self.dispatch_strategy == DispatchStrategy.BALANCED:
                driver = self.assign_order_balanced(order, available_drivers)
            else:
                driver = None

            if driver:
                # Assign order to driver
                order.assigned_time = self.current_time
                order.driver_id = driver.driver_id
                driver.current_order = order
                driver.status = DriverStatus.PICKING_UP

                # Calculate travel time to restaurant
                distance_to_restaurant = driver.current_location.distance_to(order.restaurant_location)
                travel_time = distance_to_restaurant / self.avg_speed

                # Driver will arrive at restaurant at this time
                pickup_time = self.current_time + travel_time

                # Pickup happens when both driver arrives AND food is ready
                actual_pickup_time = max(pickup_time, order.arrival_time + order.prep_time)
                order.pickup_time = actual_pickup_time

                # Calculate delivery time
                delivery_distance = order.restaurant_location.distance_to(order.customer_location)
                delivery_travel_time = delivery_distance / self.avg_speed
                order.delivery_time = actual_pickup_time + delivery_travel_time

                # Update driver
                driver.total_distance += distance_to_restaurant + delivery_distance
                driver.available_at = order.delivery_time
                driver.current_location = order.customer_location

                # Remove from available list
                available_drivers.remove(driver)
                orders_to_remove.append(order)

        # Remove assigned orders from pending
        for order in orders_to_remove:
            self.pending_orders.remove(order)

    def update_driver_status(self):
        """Update driver statuses based on current time"""
        for driver in self.drivers:
            if driver.status != DriverStatus.AVAILABLE and driver.available_at <= self.current_time:
                if driver.current_order:
                    # Order completed
                    driver.completed_orders.append(driver.current_order)
                    self.completed_orders.append(driver.current_order)

                    # Track metrics
                    if driver.current_order.total_time:
                        self.delivery_times.append(driver.current_order.total_time)
                    if driver.current_order.wait_time:
                        self.wait_times.append(driver.current_order.wait_time)

                    driver.current_order = None

                driver.status = DriverStatus.AVAILABLE

    def track_utilization(self):
        """Track driver utilization at current time"""
        for driver in self.drivers:
            is_busy = 1 if driver.status != DriverStatus.AVAILABLE else 0
            self.driver_utilization[driver.driver_id].append(is_busy)

    def simulate_step(self, dt=1.0):
        """
        Advance simulation by one time step

        Args:
            dt: Time step in minutes
        """
        self.current_time += dt
        self.update_driver_status()
        self.assign_orders()
        self.track_utilization()


def generate_orders(duration, order_rate, service_area_size, avg_prep_time=15):
    """
    Generate random orders using Poisson process

    Args:
        duration: Simulation duration in minutes
        order_rate: Average orders per minute
        service_area_size: Size of service area (km)
        avg_prep_time: Average food preparation time (minutes)

    Returns:
        List of Order objects
    """
    orders = []
    order_id = 0

    # Generate order arrival times
    t = 0
    while t < duration:
        t += np.random.exponential(1 / order_rate)
        if t < duration:
            # Random restaurant location
            restaurant = Location(
                x=np.random.uniform(0, service_area_size),
                y=np.random.uniform(0, service_area_size)
            )

            # Customer location (typically within 5km of restaurant)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0.5, 5.0)  # 0.5 to 5 km away
            customer = Location(
                x=restaurant.x + distance * np.cos(angle),
                y=restaurant.y + distance * np.sin(angle)
            )

            # Clip to service area
            customer.x = np.clip(customer.x, 0, service_area_size)
            customer.y = np.clip(customer.y, 0, service_area_size)

            # Random prep time (normal distribution)
            prep_time = max(5, np.random.normal(avg_prep_time, 5))

            order = Order(
                order_id=order_id,
                arrival_time=t,
                restaurant_location=restaurant,
                customer_location=customer,
                prep_time=prep_time
            )
            orders.append(order)
            order_id += 1

    return orders


def run_simulation(num_drivers=10, dispatch_strategy=DispatchStrategy.NEAREST, duration=480, order_rate=0.5, service_area_size=10.0, avg_speed=0.5):
    """
    Run complete delivery system simulation

    Args:
        num_drivers: Number of delivery drivers
        dispatch_strategy: Order assignment strategy
        duration: Simulation duration in minutes
        order_rate: Average orders per minute
        service_area_size: Service area size (km)
        avg_speed: Driver speed (km/minute)

    Returns:
        DeliverySystem object with results
    """
    # Generate orders
    orders = generate_orders(duration, order_rate, service_area_size)

    # Create delivery system
    system = DeliverySystem(
        num_drivers=num_drivers,
        dispatch_strategy=dispatch_strategy,
        service_area_size=service_area_size,
        avg_speed=avg_speed
    )

    # Run simulation
    order_idx = 0
    time_step = 1.0  # 1 minute steps

    for t in np.arange(0, duration, time_step):
        # Add new orders
        while order_idx < len(orders) and orders[order_idx].arrival_time <= t:
            system.add_order(orders[order_idx])
            order_idx += 1

        # Simulate one step
        system.simulate_step(dt=time_step)

    return system


def analyze_results(system: DeliverySystem, scenario_name: str) -> Dict:
    """Analyze and print simulation results"""
    completed = system.completed_orders

    if len(completed) == 0:
        print(f"\n{scenario_name}: No orders completed!")
        return {}

    wait_times = [o.wait_time for o in completed if o.wait_time is not None]
    delivery_times = [o.total_time for o in completed if o.total_time is not None]

    # Driver statistics
    driver_orders = [len(d.completed_orders) for d in system.drivers]
    driver_distances = [d.total_distance for d in system.drivers]

    # Calculate utilization
    avg_utilization = []
    for driver_id in system.driver_utilization:
        if system.driver_utilization[driver_id]:
            avg_util = np.mean(system.driver_utilization[driver_id]) * 100
            avg_utilization.append(avg_util)

    results = {
        'scenario': scenario_name,
        'strategy': system.dispatch_strategy.value,
        'num_drivers': system.num_drivers,
        'total_orders': len(completed),
        'orders_pending': len(system.pending_orders),
        'avg_wait_time': np.mean(wait_times) if wait_times else 0,
        'max_wait_time': np.max(wait_times) if wait_times else 0,
        'avg_delivery_time': np.mean(delivery_times) if delivery_times else 0,
        'max_delivery_time': np.max(delivery_times) if delivery_times else 0,
        'avg_orders_per_driver': np.mean(driver_orders),
        'std_orders_per_driver': np.std(driver_orders),
        'avg_distance_per_driver': np.mean(driver_distances),
        'avg_driver_utilization': np.mean(avg_utilization) if avg_utilization else 0,
        'completion_rate': len(completed) / system.total_orders_received * 100
    }

    print(f"\n{'=' * 70}")
    print(f"Results for: {scenario_name}")
    print(f"{'=' * 70}")
    print(f"Strategy: {system.dispatch_strategy.value.upper()}")
    print(f"Number of drivers: {system.num_drivers}")
    print(f"Total orders received: {system.total_orders_received}")
    print(f"Orders completed: {len(completed)}")
    print(f"Orders pending: {len(system.pending_orders)}")
    print(f"Completion rate: {results['completion_rate']:.1f}%")
    print(f"\n--- Time Metrics ---")
    print(f"Average wait time: {results['avg_wait_time']:.2f} minutes")
    print(f"Maximum wait time: {results['max_wait_time']:.2f} minutes")
    print(f"Average delivery time: {results['avg_delivery_time']:.2f} minutes")
    print(f"Maximum delivery time: {results['max_delivery_time']:.2f} minutes")
    print(f"\n--- Driver Metrics ---")
    print(f"Average orders per driver: {results['avg_orders_per_driver']:.1f}")
    print(f"Std dev orders per driver: {results['std_orders_per_driver']:.2f}")
    print(f"Average distance per driver: {results['avg_distance_per_driver']:.2f} km")
    print(f"Average driver utilization: {results['avg_driver_utilization']:.1f}%")

    return results


def visualize_comparison(all_results: List[Dict]):
    """Create comprehensive comparison visualizations"""
    df = pd.DataFrame(all_results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Food Delivery System Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Average Delivery Time
    ax1 = axes[0, 0]
    x_pos = np.arange(len(df))
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D', '#A8E6CF'][:len(df)]
    ax1.bar(x_pos, df['avg_delivery_time'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Scenario', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Average Delivery Time (minutes)', fontweight='bold', fontsize=11)
    ax1.set_title('Average Total Delivery Time', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['scenario'], rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Wait Time vs Delivery Time
    ax2 = axes[0, 1]
    x_pos = np.arange(len(df))
    width = 0.35
    ax2.bar(x_pos - width / 2, df['avg_wait_time'], width, label='Wait Time', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos + width / 2, df['avg_delivery_time'], width, label='Total Delivery Time', color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Scenario', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Time (minutes)', fontweight='bold', fontsize=11)
    ax2.set_title('Wait Time vs Total Delivery Time', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df['scenario'], rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Driver Utilization
    ax3 = axes[1, 0]
    ax3.bar(x_pos, df['avg_driver_utilization'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Scenario', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Utilization (%)', fontweight='bold', fontsize=11)
    ax3.set_title('Average Driver Utilization', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df['scenario'], rotation=45, ha='right', fontsize=9)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Target: 70%')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Workload Balance
    ax4 = axes[1, 1]
    ax4.bar(x_pos, df['std_orders_per_driver'], color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Scenario', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Standard Deviation', fontweight='bold', fontsize=11)
    ax4.set_title('Workload Balance (Lower is Better)', fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df['scenario'], rotation=45, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('delivery_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: delivery_comparison.png")
    plt.show()


def plot_driver_distribution(system: DeliverySystem, scenario_name: str):
    """Plot distribution of orders across drivers"""
    orders_per_driver = [len(d.completed_orders) for d in system.drivers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Driver Workload Distribution: {scenario_name}', fontsize=14, fontweight='bold')

    # Bar chart
    driver_ids = [d.driver_id for d in system.drivers]
    colors = plt.cm.viridis(np.linspace(0, 1, len(driver_ids)))
    ax1.bar(driver_ids, orders_per_driver, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Driver ID', fontweight='bold')
    ax1.set_ylabel('Orders Completed', fontweight='bold')
    ax1.set_title('Orders per Driver', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Distance traveled
    distances = [d.total_distance for d in system.drivers]
    ax2.bar(driver_ids, distances, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Driver ID', fontweight='bold')
    ax2.set_ylabel('Total Distance (km)', fontweight='bold')
    ax2.set_title('Distance Traveled per Driver', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filename = f'driver_distribution_{scenario_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {filename}")
    plt.show()


# =============================================================================
# MAIN SIMULATION SCENARIOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FOOD DELIVERY DISPATCH SYSTEM SIMULATION")
    print("=" * 70)

    np.random.seed(42)  # For reproducibility

    # Simulation parameters
    duration = 480  # 8 hours (lunch + dinner rush)
    service_area = 10.0  # 10km x 10km area

    # Scenario 1: Normal Demand - Nearest Assignment
    print("\n\nRunning Scenario 1: Normal Demand with Nearest Driver Strategy...")
    sim1 = run_simulation(
        num_drivers=10,
        dispatch_strategy=DispatchStrategy.NEAREST,
        duration=duration,
        order_rate=0.4,  # 24 orders/hour
        service_area_size=service_area,
        avg_speed=0.5  # 30 km/hour
    )
    results1 = analyze_results(sim1, "Scenario 1: Normal - Nearest")
    plot_driver_distribution(sim1, "Scenario 1")

    # Scenario 2: Normal Demand - FCFS
    print("\n\nRunning Scenario 2: Normal Demand with FCFS Strategy...")
    sim2 = run_simulation(
        num_drivers=10,
        dispatch_strategy=DispatchStrategy.FCFS,
        duration=duration,
        order_rate=0.4,
        service_area_size=service_area,
        avg_speed=0.5
    )
    results2 = analyze_results(sim2, "Scenario 2: Normal - FCFS")

    # Scenario 3: Normal Demand - Balanced
    print("\n\nRunning Scenario 3: Normal Demand with Balanced Strategy...")
    sim3 = run_simulation(
        num_drivers=10,
        dispatch_strategy=DispatchStrategy.BALANCED,
        duration=duration,
        order_rate=0.4,
        service_area_size=service_area,
        avg_speed=0.5
    )
    results3 = analyze_results(sim3, "Scenario 3: Normal - Balanced")

    # Scenario 4: Peak Demand - Nearest (understaffed)
    print("\n\nRunning Scenario 4: Peak Demand with Limited Drivers...")
    sim4 = run_simulation(
        num_drivers=10,
        dispatch_strategy=DispatchStrategy.NEAREST,
        duration=duration,
        order_rate=0.7,  # 42 orders/hour - high demand
        service_area_size=service_area,
        avg_speed=0.5
    )
    results4 = analyze_results(sim4, "Scenario 4: Peak - 10 Drivers")

    # Scenario 5: Peak Demand - Nearest (well-staffed)
    print("\n\nRunning Scenario 5: Peak Demand with Adequate Drivers...")
    sim5 = run_simulation(
        num_drivers=15,
        dispatch_strategy=DispatchStrategy.NEAREST,
        duration=duration,
        order_rate=0.7,
        service_area_size=service_area,
        avg_speed=0.5
    )
    results5 = analyze_results(sim5, "Scenario 5: Peak - 15 Drivers")
    plot_driver_distribution(sim5, "Scenario 5")

    # Create comprehensive comparison
    all_results = [results1, results2, results3, results4, results5]
    visualize_comparison(all_results)

    # Summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    df_summary = pd.DataFrame(all_results)
    print("\n", df_summary[['scenario', 'strategy', 'avg_delivery_time', 'avg_driver_utilization', 'completion_rate']].to_string(index=False))

    print("\n✓ Simulation complete! Check the generated PNG files for visualizations.")
    print("\nKey Insights:")
    print("- Compare dispatch strategies under normal demand (Scenarios 1-3)")
    print("- Analyze impact of driver availability during peak hours (Scenarios 4-5)")
    print("- Evaluate workload balance across different strategies")
