#!/usr/bin/env python3

####################################################################################
#                                                                                  #
#    ██████╗ ███████╗████████╗   ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗   #
#   ██╔════╝ ██╔════╝╚══██╔══╝   ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝   #
#   ██║  ███╗█████╗     ██║      ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝    #
#   ██║   ██║██╔══╝     ██║      ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝     #
#   ╚██████╔╝███████╗   ██║      ███████║╚███╔███╔╝██║██╗        ██║      ██║      #
#    ╚═════╝ ╚══════╝   ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝      #
#                                                                                  #
####################################################################################
#
# Script Name: forensics-traffic-analyzer.py                                     
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced network traffic analysis tool for forensic investigation
#              with packet inspection, protocol analysis, flow reconstruction,  
#              and anomaly detection for comprehensive network forensics.       
#
# Usage: python forensics-traffic-analyzer.py [--input FILE] [--output DIR]     
#
# Dependencies: scapy, pyshark, pandas, matplotlib, networkx                   
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional network forensics with deep packet inspection, session   
#        reconstruction, and comprehensive analysis for investigations.         
#                                                                                
####################################################################################

"""
Advanced Forensic Network Traffic Analyzer
==========================================

Comprehensive network traffic analysis tool for digital forensics investigations.
Performs packet inspection, protocol analysis, flow reconstruction, and anomaly
detection with detailed reporting and visualization capabilities.
"""

import os
import sys
import logging
import multiprocessing
import subprocess
import argparse
import time
import json
import socket
import struct
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "scapy>=2.4.5",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "networkx>=2.6.0"
    ]
    
    missing = []
    for pkg in required_packages:
        name = pkg.split('>=')[0].replace('-', '_')
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

check_and_install_dependencies()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import Ether, ARP
    from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

IS_MACOS = sys.platform == "darwin"

class ForensicTrafficAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        # Analysis statistics
        self.analysis_stats = {
            'total_packets': 0,
            'processed_packets': 0,
            'unique_flows': 0,
            'suspicious_activities': 0,
            'protocols_detected': set(),
            'start_time': None,
            'duration': 0
        }
        
        # Traffic data storage
        self.packets = []
        self.flows = defaultdict(list)
        self.connections = defaultdict(dict)
        self.dns_queries = []
        self.http_requests = []
        self.suspicious_activities = []
        
        # Analysis results
        self.analysis_results = {}
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "forensics-traffic-analyzer.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
    def select_pcap_file(self):
        """Select PCAP file via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        file_path = filedialog.askopenfilename(
            title="Select PCAP File for Analysis",
            filetypes=[
                ("PCAP files", "*.pcap;*.pcapng;*.cap"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return Path(file_path) if file_path else None
        
    def select_output_directory(self):
        """Select output directory via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        directory = filedialog.askdirectory(
            title="Select Output Directory for Analysis Reports"
        )
        
        root.destroy()
        return Path(directory) if directory else None
        
    def create_progress_window(self, total_packets):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Network Traffic Analysis")
        self.progress_root.geometry("500x150")
        self.progress_root.resizable(False, False)
        
        # Center window
        x = (self.progress_root.winfo_screenwidth() // 2) - 250
        y = (self.progress_root.winfo_screenheight() // 2) - 75
        self.progress_root.geometry(f"+{x}+{y}")
        
        if IS_MACOS:
            self.progress_root.attributes("-topmost", True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_root,
            variable=self.progress_var,
            maximum=total_packets,
            length=450
        )
        self.progress_bar.pack(pady=20)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing traffic analysis...")
        self.status_label = tk.Label(
            self.progress_root,
            textvariable=self.status_var,
            font=("Helvetica", 11)
        )
        self.status_label.pack(pady=(0, 20))
        
        self.progress_root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.progress_root.update()
        
    def update_progress(self, current, total, status=""):
        """Update progress bar and status"""
        if hasattr(self, 'progress_var'):
            self.progress_var.set(current)
            
            percentage = int((current / total) * 100)
            if status:
                self.status_var.set(f"{status} ({percentage}%)")
            else:
                self.status_var.set(f"Processing {current}/{total} ({percentage}%)")
                
            self.progress_root.update()
            
    def extract_packet_info(self, packet):
        """Extract comprehensive information from a packet"""
        packet_info = {
            'timestamp': float(packet.time),
            'size': len(packet),
            'protocols': [],
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None,
            'flags': [],
            'payload_size': 0
        }
        
        try:
            # Ethernet layer
            if packet.haslayer(Ether):
                packet_info['src_mac'] = packet[Ether].src
                packet_info['dst_mac'] = packet[Ether].dst
                packet_info['protocols'].append('Ethernet')
                
            # IP layer (IPv4)
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                packet_info['src_ip'] = ip_layer.src
                packet_info['dst_ip'] = ip_layer.dst
                packet_info['protocol'] = ip_layer.proto
                packet_info['ttl'] = ip_layer.ttl
                packet_info['protocols'].append('IPv4')
                
            # IPv6 layer
            elif packet.haslayer(IPv6):
                ipv6_layer = packet[IPv6]
                packet_info['src_ip'] = ipv6_layer.src
                packet_info['dst_ip'] = ipv6_layer.dst
                packet_info['protocol'] = ipv6_layer.nh
                packet_info['protocols'].append('IPv6')
                
            # TCP layer
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                packet_info['src_port'] = tcp_layer.sport
                packet_info['dst_port'] = tcp_layer.dport
                packet_info['seq_num'] = tcp_layer.seq
                packet_info['ack_num'] = tcp_layer.ack
                
                # TCP flags
                flags = []
                if tcp_layer.flags.S: flags.append('SYN')
                if tcp_layer.flags.A: flags.append('ACK')
                if tcp_layer.flags.F: flags.append('FIN')
                if tcp_layer.flags.R: flags.append('RST')
                if tcp_layer.flags.P: flags.append('PSH')
                if tcp_layer.flags.U: flags.append('URG')
                
                packet_info['flags'] = flags
                packet_info['protocols'].append('TCP')
                
            # UDP layer
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                packet_info['src_port'] = udp_layer.sport
                packet_info['dst_port'] = udp_layer.dport
                packet_info['protocols'].append('UDP')
                
            # ICMP layer
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                packet_info['icmp_type'] = icmp_layer.type
                packet_info['icmp_code'] = icmp_layer.code
                packet_info['protocols'].append('ICMP')
                
            # ARP layer
            if packet.haslayer(ARP):
                arp_layer = packet[ARP]
                packet_info['arp_op'] = arp_layer.op
                packet_info['arp_hwsrc'] = arp_layer.hwsrc
                packet_info['arp_hwdst'] = arp_layer.hwdst
                packet_info['protocols'].append('ARP')
                
            # DNS layer
            if packet.haslayer(DNS):
                dns_layer = packet[DNS]
                packet_info['dns_id'] = dns_layer.id
                packet_info['dns_qr'] = dns_layer.qr
                packet_info['protocols'].append('DNS')
                
                # DNS queries
                if dns_layer.qd:
                    queries = []
                    for i in range(dns_layer.qdcount):
                        if i < len(dns_layer.qd):
                            query = dns_layer.qd[i]
                            queries.append({
                                'name': query.qname.decode('utf-8', errors='ignore'),
                                'type': query.qtype,
                                'class': query.qclass
                            })
                    packet_info['dns_queries'] = queries
                    
            # HTTP layer
            if packet.haslayer(HTTPRequest):
                http_req = packet[HTTPRequest]
                packet_info['http_method'] = http_req.Method.decode('utf-8', errors='ignore')
                packet_info['http_host'] = http_req.Host.decode('utf-8', errors='ignore') if http_req.Host else None
                packet_info['http_path'] = http_req.Path.decode('utf-8', errors='ignore') if http_req.Path else None
                packet_info['protocols'].append('HTTP')
                
            # Calculate payload size
            if packet.haslayer(Raw):
                packet_info['payload_size'] = len(packet[Raw])
                
        except Exception as e:
            self.logger.warning(f"Error extracting packet info: {e}")
            
        return packet_info
        
    def analyze_packet(self, packet):
        """Analyze individual packet for anomalies and extract metadata"""
        packet_info = self.extract_packet_info(packet)
        
        # Update statistics
        self.analysis_stats['processed_packets'] += 1
        for protocol in packet_info['protocols']:
            self.analysis_stats['protocols_detected'].add(protocol)
            
        # Flow tracking
        if packet_info['src_ip'] and packet_info['dst_ip']:
            flow_key = self.create_flow_key(packet_info)
            self.flows[flow_key].append(packet_info)
            
        # Store DNS queries
        if 'dns_queries' in packet_info:
            for query in packet_info['dns_queries']:
                query['timestamp'] = packet_info['timestamp']
                query['src_ip'] = packet_info['src_ip']
                self.dns_queries.append(query)
                
        # Store HTTP requests
        if 'http_method' in packet_info:
            http_request = {
                'timestamp': packet_info['timestamp'],
                'src_ip': packet_info['src_ip'],
                'dst_ip': packet_info['dst_ip'],
                'method': packet_info['http_method'],
                'host': packet_info.get('http_host'),
                'path': packet_info.get('http_path')
            }
            self.http_requests.append(http_request)
            
        # Anomaly detection
        self.detect_packet_anomalies(packet_info)
        
        return packet_info
        
    def create_flow_key(self, packet_info):
        """Create a unique flow identifier"""
        src_ip = packet_info['src_ip']
        dst_ip = packet_info['dst_ip']
        src_port = packet_info.get('src_port', 0)
        dst_port = packet_info.get('dst_port', 0)
        protocol = packet_info.get('protocol', 0)
        
        # Normalize flow direction (smaller IP first)
        if src_ip < dst_ip:
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
            
    def detect_packet_anomalies(self, packet_info):
        """Detect anomalies in individual packets"""
        anomalies = []
        
        # Port scanning detection
        if packet_info.get('dst_port'):
            dst_port = packet_info['dst_port']
            
            # Common suspicious ports
            suspicious_ports = {
                23: 'Telnet', 135: 'RPC', 139: 'NetBIOS', 445: 'SMB',
                1433: 'SQL Server', 3389: 'RDP', 5900: 'VNC'
            }
            
            if dst_port in suspicious_ports:
                anomalies.append(f"Connection to suspicious port {dst_port} ({suspicious_ports[dst_port]})")
                
        # Unusual packet sizes
        if packet_info['size'] > 9000:  # Jumbo frames
            anomalies.append(f"Unusually large packet: {packet_info['size']} bytes")
        elif packet_info['size'] < 64 and 'TCP' in packet_info['protocols']:
            anomalies.append(f"Unusually small TCP packet: {packet_info['size']} bytes")
            
        # DNS anomalies
        if 'dns_queries' in packet_info:
            for query in packet_info['dns_queries']:
                domain = query['name']
                if len(domain) > 100:
                    anomalies.append(f"Suspiciously long domain name: {domain[:50]}...")
                if domain.count('.') > 10:
                    anomalies.append(f"Domain with excessive subdomains: {domain}")
                    
        # HTTP anomalies
        if 'http_method' in packet_info:
            method = packet_info['http_method']
            if method not in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']:
                anomalies.append(f"Unusual HTTP method: {method}")
                
        # Private IP communication with internet
        if packet_info['src_ip'] and packet_info['dst_ip']:
            src_private = self.is_private_ip(packet_info['src_ip'])
            dst_private = self.is_private_ip(packet_info['dst_ip'])
            
            if src_private and not dst_private:
                # Outbound traffic from private to public
                pass  # Normal
            elif not src_private and dst_private:
                anomalies.append("Inbound traffic from public to private IP")
                
        if anomalies:
            suspicious_activity = {
                'timestamp': packet_info['timestamp'],
                'src_ip': packet_info['src_ip'],
                'dst_ip': packet_info['dst_ip'],
                'anomalies': anomalies,
                'packet_info': packet_info
            }
            self.suspicious_activities.append(suspicious_activity)
            self.analysis_stats['suspicious_activities'] += 1
            
    def is_private_ip(self, ip_str):
        """Check if IP address is in private ranges"""
        try:
            ip_parts = [int(x) for x in ip_str.split('.')]
            
            # Private IP ranges:
            # 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            if ip_parts[0] == 10:
                return True
            elif ip_parts[0] == 172 and 16 <= ip_parts[1] <= 31:
                return True
            elif ip_parts[0] == 192 and ip_parts[1] == 168:
                return True
                
        except:
            pass
            
        return False
        
    def analyze_flows(self):
        """Analyze network flows for patterns and anomalies"""
        flow_analysis = {}
        
        self.analysis_stats['unique_flows'] = len(self.flows)
        
        for flow_key, packets in self.flows.items():
            if len(packets) < 2:
                continue
                
            # Flow statistics
            flow_stats = {
                'packet_count': len(packets),
                'total_bytes': sum(p['size'] for p in packets),
                'duration': packets[-1]['timestamp'] - packets[0]['timestamp'],
                'avg_packet_size': statistics.mean(p['size'] for p in packets),
                'protocols': list(set(p for packet in packets for p in packet['protocols']))
            }
            
            # Direction analysis
            directions = defaultdict(int)
            for packet in packets:
                direction = f"{packet['src_ip']}>{packet['dst_ip']}"
                directions[direction] += 1
                
            flow_stats['directions'] = dict(directions)
            
            # Temporal analysis
            intervals = []
            for i in range(1, len(packets)):
                interval = packets[i]['timestamp'] - packets[i-1]['timestamp']
                intervals.append(interval)
                
            if intervals:
                flow_stats['avg_interval'] = statistics.mean(intervals)
                flow_stats['interval_std'] = statistics.stdev(intervals) if len(intervals) > 1 else 0
                
            # Anomaly detection for flows
            anomalies = []
            
            # Long-duration flows
            if flow_stats['duration'] > 3600:  # More than 1 hour
                anomalies.append(f"Long-duration flow: {flow_stats['duration']:.1f} seconds")
                
            # High packet rate
            if flow_stats['duration'] > 0:
                packet_rate = flow_stats['packet_count'] / flow_stats['duration']
                if packet_rate > 100:  # More than 100 packets per second
                    anomalies.append(f"High packet rate: {packet_rate:.1f} pps")
                    
            # Large data transfer
            if flow_stats['total_bytes'] > 100 * 1024 * 1024:  # More than 100MB
                anomalies.append(f"Large data transfer: {flow_stats['total_bytes'] / (1024*1024):.1f} MB")
                
            flow_stats['anomalies'] = anomalies
            flow_analysis[flow_key] = flow_stats
            
        return flow_analysis
        
    def analyze_protocols(self):
        """Analyze protocol distribution and patterns"""
        protocol_analysis = {}
        
        # Protocol distribution
        protocol_counts = defaultdict(int)
        for packet in self.packets:
            for protocol in packet.get('protocols', []):
                protocol_counts[protocol] += 1
                
        protocol_analysis['distribution'] = dict(protocol_counts)
        
        # Port analysis
        port_counts = defaultdict(int)
        for packet in self.packets:
            if packet.get('dst_port'):
                port_counts[packet['dst_port']] += 1
                
        # Top ports
        top_ports = sorted(port_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        protocol_analysis['top_destination_ports'] = top_ports
        
        # Service identification
        common_ports = {
            80: 'HTTP', 443: 'HTTPS', 53: 'DNS', 21: 'FTP',
            22: 'SSH', 23: 'Telnet', 25: 'SMTP', 110: 'POP3',
            143: 'IMAP', 993: 'IMAPS', 995: 'POP3S'
        }
        
        services = defaultdict(int)
        for port, count in port_counts.items():
            service = common_ports.get(port, f'Port-{port}')
            services[service] += count
            
        protocol_analysis['services'] = dict(services)
        
        return protocol_analysis
        
    def analyze_dns_activity(self):
        """Analyze DNS queries for suspicious patterns"""
        dns_analysis = {}
        
        if not self.dns_queries:
            return dns_analysis
            
        # Domain frequency
        domain_counts = defaultdict(int)
        for query in self.dns_queries:
            domain = query['name'].rstrip('.')
            domain_counts[domain] += 1
            
        # Top queried domains
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        dns_analysis['top_domains'] = top_domains
        
        # Suspicious domain patterns
        suspicious_domains = []
        for domain, count in domain_counts.items():
            # Very long domains
            if len(domain) > 50:
                suspicious_domains.append(f"Long domain: {domain}")
                
            # Domains with many subdomains
            if domain.count('.') > 8:
                suspicious_domains.append(f"Many subdomains: {domain}")
                
            # Random-looking domains
            if self.looks_like_dga(domain):
                suspicious_domains.append(f"Possible DGA domain: {domain}")
                
        dns_analysis['suspicious_domains'] = suspicious_domains
        
        # Query type analysis
        query_types = defaultdict(int)
        for query in self.dns_queries:
            query_types[query['type']] += 1
            
        dns_analysis['query_types'] = dict(query_types)
        
        return dns_analysis
        
    def looks_like_dga(self, domain):
        """Simple heuristic to detect Domain Generation Algorithm (DGA) domains"""
        # Remove TLD
        domain_parts = domain.split('.')
        if len(domain_parts) < 2:
            return False
            
        main_domain = domain_parts[-2]
        
        # Check for randomness indicators
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        
        vowel_count = sum(1 for c in main_domain.lower() if c in vowels)
        consonant_count = sum(1 for c in main_domain.lower() if c in consonants)
        
        # Heuristics for DGA detection
        if len(main_domain) > 12:  # Long domains
            if vowel_count < 2:  # Very few vowels
                return True
            if consonant_count > vowel_count * 3:  # Too many consonants
                return True
                
        return False
        
    def generate_analysis_report(self, output_dir):
        """Generate comprehensive traffic analysis report"""
        try:
            # Create reports directory
            reports_dir = output_dir / "traffic_analysis_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Perform detailed analysis
            flow_analysis = self.analyze_flows()
            protocol_analysis = self.analyze_protocols()
            dns_analysis = self.analyze_dns_activity()
            
            # Compile results
            self.analysis_results = {
                'summary': dict(self.analysis_stats),
                'flows': flow_analysis,
                'protocols': protocol_analysis,
                'dns': dns_analysis,
                'suspicious_activities': self.suspicious_activities,
                'http_requests': self.http_requests[:100]  # Limit to first 100
            }
            
            # Convert sets to lists for JSON serialization
            self.analysis_results['summary']['protocols_detected'] = list(self.analysis_stats['protocols_detected'])
            
            # Generate JSON report
            json_report = reports_dir / f"traffic_analysis_{timestamp}.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate HTML report
            html_report = reports_dir / f"traffic_analysis_{timestamp}.html"
            self.generate_html_report(html_report)
            
            # Generate network graph
            graph_file = reports_dir / f"network_graph_{timestamp}.png"
            self.generate_network_graph(graph_file)
            
            self.logger.info(f"✓ Reports generated in: {reports_dir}")
            return reports_dir
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
            
    def generate_html_report(self, html_path):
        """Generate HTML traffic analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Network Traffic Analysis Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 40px; background: #0a0a0a; color: #00ff00; }}
        .header {{ background: #1a1a1a; padding: 20px; border: 2px solid #00ff00; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #1a1a1a; padding: 15px; border: 1px solid #00ff00; text-align: center; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #333; background: #1a1a1a; }}
        .suspicious {{ color: #ff4444; background: #2a1a1a; padding: 10px; margin: 10px 0; }}
        .protocols {{ color: #aaaaaa; }}
        h1, h2, h3 {{ color: #00ff00; }}
        .flow {{ margin: 10px 0; padding: 10px; border-left: 3px solid #00ff00; }}
        .anomaly {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌐 NETWORK TRAFFIC ANALYSIS REPORT</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analyzer: Forensic Traffic Analyzer v1.0.0</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{self.analysis_stats['processed_packets']:,}</h3>
            <p>Packets Analyzed</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['unique_flows']}</h3>
            <p>Unique Flows</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['suspicious_activities']}</h3>
            <p>Suspicious Activities</p>
        </div>
        <div class="metric">
            <h3>{len(self.analysis_stats['protocols_detected'])}</h3>
            <p>Protocols Detected</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Protocol Distribution</h2>
        <div class="protocols">
"""
        
        if 'protocols' in self.analysis_results:
            protocol_dist = self.analysis_results['protocols'].get('distribution', {})
            for protocol, count in sorted(protocol_dist.items(), key=lambda x: x[1], reverse=True):
                html_content += f"            <p>{protocol}: {count:,} packets</p>\n"
                
        html_content += """        </div>
    </div>
"""
        
        # Top services section
        if 'protocols' in self.analysis_results and 'services' in self.analysis_results['protocols']:
            html_content += """
    <div class="section">
        <h2>🔧 Top Services</h2>
        <div class="protocols">
"""
            services = self.analysis_results['protocols']['services']
            for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True)[:10]:
                html_content += f"            <p>{service}: {count:,} connections</p>\n"
                
            html_content += "        </div>\n    </div>\n"
        
        # DNS analysis section
        if 'dns' in self.analysis_results and self.analysis_results['dns']:
            dns_data = self.analysis_results['dns']
            html_content += """
    <div class="section">
        <h2>🔍 DNS Analysis</h2>
"""
            if 'top_domains' in dns_data:
                html_content += "        <h3>Top Queried Domains</h3>\n        <div class=\"protocols\">\n"
                for domain, count in dns_data['top_domains'][:10]:
                    html_content += f"            <p>{domain}: {count} queries</p>\n"
                html_content += "        </div>\n"
                
            if 'suspicious_domains' in dns_data and dns_data['suspicious_domains']:
                html_content += """        <h3>Suspicious Domains</h3>
        <div class="suspicious">
            <ul>
"""
                for domain in dns_data['suspicious_domains'][:10]:
                    html_content += f"                <li>{domain}</li>\n"
                html_content += "            </ul>\n        </div>\n"
                
            html_content += "    </div>\n"
        
        # Suspicious activities section
        if self.suspicious_activities:
            html_content += """
    <div class="section">
        <h2>🚨 Suspicious Activities</h2>
"""
            for activity in self.suspicious_activities[:20]:  # Show first 20
                timestamp = datetime.fromtimestamp(activity['timestamp']).strftime('%H:%M:%S')
                html_content += f"""
        <div class="suspicious">
            <h4>{timestamp} - {activity['src_ip']} → {activity['dst_ip']}</h4>
            <ul>
"""
                for anomaly in activity['anomalies']:
                    html_content += f"                <li>{anomaly}</li>\n"
                html_content += "            </ul>\n        </div>\n"
                
            html_content += "    </div>\n"
        
        # Flow analysis section
        if 'flows' in self.analysis_results:
            flows_with_anomalies = {k: v for k, v in self.analysis_results['flows'].items() if v.get('anomalies')}
            if flows_with_anomalies:
                html_content += """
    <div class="section">
        <h2>🔄 Anomalous Flows</h2>
"""
                for flow_key, flow_data in list(flows_with_anomalies.items())[:10]:
                    html_content += f"""
        <div class="flow">
            <h4>{flow_key}</h4>
            <p>Packets: {flow_data['packet_count']}, Bytes: {flow_data['total_bytes']:,}</p>
            <div class="anomaly">
                <ul>
"""
                    for anomaly in flow_data['anomalies']:
                        html_content += f"                    <li>{anomaly}</li>\n"
                    html_content += "                </ul>\n            </div>\n        </div>\n"
                    
                html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def generate_network_graph(self, graph_file):
        """Generate network topology graph"""
        try:
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes and edges from flows
            for flow_key, packets in list(self.flows.items())[:100]:  # Limit for performance
                if not packets:
                    continue
                    
                src_ip = packets[0]['src_ip']
                dst_ip = packets[0]['dst_ip']
                
                if src_ip and dst_ip:
                    # Add nodes
                    G.add_node(src_ip)
                    G.add_node(dst_ip)
                    
                    # Add edge with weight based on packet count
                    if G.has_edge(src_ip, dst_ip):
                        G[src_ip][dst_ip]['weight'] += len(packets)
                    else:
                        G.add_edge(src_ip, dst_ip, weight=len(packets))
                        
            if G.number_of_nodes() > 0:
                plt.figure(figsize=(12, 8))
                plt.style.use('dark_background')
                
                # Calculate layout
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Draw nodes
                node_colors = ['red' if any(activity['src_ip'] == node or activity['dst_ip'] == node 
                                          for activity in self.suspicious_activities) 
                             else 'lightblue' for node in G.nodes()]
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     node_size=100, alpha=0.8)
                
                # Draw edges
                edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                max_weight = max(edge_weights) if edge_weights else 1
                edge_widths = [w / max_weight * 3 for w in edge_weights]
                
                nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                     alpha=0.6, edge_color='gray')
                
                # Draw labels for important nodes
                important_nodes = {node: node for node in G.nodes() 
                                 if G.degree(node) > 5}  # High-degree nodes
                nx.draw_networkx_labels(G, pos, important_nodes, 
                                      font_size=8, font_color='white')
                
                plt.title('Network Topology and Traffic Flows', 
                         color='white', fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(graph_file, dpi=300, bbox_inches='tight',
                           facecolor='black', edgecolor='none')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Network graph generation error: {e}")
            
    def show_completion_summary(self, reports_dir):
        """Show analysis completion summary"""
        try:
            elapsed = time.time() - self.analysis_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""Network Traffic Analysis Completed!

Packets Analyzed: {self.analysis_stats['processed_packets']:,}
Unique Flows: {self.analysis_stats['unique_flows']}
Suspicious Activities: {self.analysis_stats['suspicious_activities']}
Protocols Detected: {len(self.analysis_stats['protocols_detected'])}

Time Elapsed: {int(minutes)}m {int(seconds)}s

Reports saved to: {reports_dir.name}

Check the log file for detailed information."""

            root = tk.Tk()
            root.withdraw()
            
            if IS_MACOS:
                root.attributes("-topmost", True)
                
            messagebox.showinfo("Analysis Complete", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Advanced Forensic Network Traffic Analyzer")
        parser.add_argument('--input', help='Input PCAP file')
        parser.add_argument('--output', help='Output directory for reports')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Forensic Traffic Analyzer v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        if not SCAPY_AVAILABLE:
            self.logger.error("Scapy is not available. Please install with: pip install scapy")
            return
            
        self.logger.info("Starting network traffic analysis...")
        
        try:
            # Get input PCAP file
            if args.input:
                pcap_file = Path(args.input)
                if not pcap_file.exists():
                    self.logger.error(f"PCAP file not found: {args.input}")
                    return
            else:
                pcap_file = self.select_pcap_file()
                
            if not pcap_file:
                self.logger.warning("No PCAP file selected")
                return
                
            # Get output directory
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = self.select_output_directory()
                
            if not output_dir:
                self.logger.error("No output directory selected")
                return
                
            self.analysis_stats['start_time'] = time.time()
            
            self.logger.info(f"Loading PCAP file: {pcap_file}")
            
            # Load packets
            try:
                packets = rdpcap(str(pcap_file))
                self.analysis_stats['total_packets'] = len(packets)
                
                self.logger.info(f"Loaded {len(packets)} packets")
                
                # Create progress window
                self.create_progress_window(len(packets))
                
                # Process packets
                for i, packet in enumerate(packets):
                    packet_info = self.analyze_packet(packet)
                    self.packets.append(packet_info)
                    
                    if i % 1000 == 0:  # Update progress every 1000 packets
                        self.update_progress(i + 1, len(packets), "Processing packets")
                        
                # Final progress update
                self.update_progress(len(packets), len(packets), "Analysis complete")
                
                # Close progress window
                if hasattr(self, 'progress_root'):
                    self.progress_root.destroy()
                    
                # Generate reports
                reports_dir = self.generate_analysis_report(output_dir)
                
                if reports_dir:
                    # Show completion summary
                    self.show_completion_summary(reports_dir)
                    
                    # Open reports directory if on macOS
                    if IS_MACOS:
                        subprocess.run(['open', str(reports_dir)], check=False)
                        
            except Exception as e:
                self.logger.error(f"Error processing PCAP file: {e}")
                if hasattr(self, 'progress_root'):
                    self.progress_root.destroy()
                return
            
        except KeyboardInterrupt:
            self.logger.info("Analysis interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Open log file
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    analyzer = ForensicTrafficAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()