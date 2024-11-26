#!/bin/bash

# Runs a PTP master reference clock, for other devices to synchronize to.
# To be run on Linux, e.g. Raspberry Pi 4.
# This script assumes that the device is connected via WiFi for internet
# and Ethernet for PTP timing.

# INSTRUCTIONS:

# Make the script executable:
#   chmod +x launchRefClock.sh

# Run like this:
#   ./launchRefClock.sh

# Ensure that ptpd is installed before proceeding.


# Check for an active internet connection via WLAN
echo -n "Checking internet connection via WiFi... "
if ! ping -I wlan0 -c 1 8.8.8.8 &> /dev/null; then
  echo "No internet connection via WiFi. Please check your wireless connection."
  exit 1
fi
echo "Internet connection verified."

# Check if ptpd is installed
echo -n "Checking if ptpd is installed... "
if ! command -v ptpd &> /dev/null; then
  echo ""
  echo "ptpd is not installed. Would you like to install it now? [y/n]"
  read -r install_ptpd
  if [[ "$install_ptpd" == "y" || "$install_ptpd" == "Y" ]]; then
    echo "Installing ptpd..."
    sudo apt update
    sudo apt install -y ptpd

    # Refresh PATH
    export PATH=$PATH:/usr/sbin:/usr/local/sbin
    hash -r

    # Verify installation
    if command -v ptpd &> /dev/null; then
      echo "ptpd installed successfully at $(command -v ptpd)"
    else
      echo "ptpd installation failed or not found in PATH."
      exit 1
    fi
  else
    echo "ptpd is required to run this script. Exiting."
    exit 1
  fi
else
  echo "ptpd is already installed."
fi

# Sync system clock to NTP
echo -n "Syncing system clock with NTP... "
if ! sudo timedatectl set-ntp true; then
  echo "Failed to enable NTP synchronization."
  exit 1
fi
echo "System clock synchronized with NTP."

# Check if eth0 is up
echo -n "Checking if Ethernet interface eth0 is up... "
if ! sudo ip link show eth0 | grep -q "state UP"; then
  echo "Ethernet interface eth0 is down. Please check your Ethernet connection."
  exit 1
fi
echo "Ethernet interface eth0 is up."

# Set IP address for PTP network
IP_ADDRESS="169.254.178.87"
echo -n "Setting PTP network IP address to $IP_ADDRESS... "
if ! sudo ifconfig eth0 $IP_ADDRESS netmask 255.255.255.0 up; then
  echo "Failed to set IP address."
  exit 1
fi
sleep 2 # Short delay to ensure network settings are applied
echo "PTP network IP address configured."

# Start ptpd command and keep it open
echo "Starting ptpd..."
echo "Press Ctrl+C to stop ptpd when needed."
sudo ptpd -M -i eth0 -a 1 -d E2E -C