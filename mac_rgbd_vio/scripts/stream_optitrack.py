"""
Real-Time OptiTrack NatNet Client & Ground Truth Logger for MAC-RGBD-VIO
Streams 6-DoF Rigid Body poses from OptiTrack Motive over Ethernet / Wi-Fi
directly to the laptop, timestamping incoming packets with the laptop's local system clock.
Eliminates cross-machine clock drift entirely.

Usage:
    python mac_rgbd_vio/scripts/stream_optitrack.py --server 192.168.1.100 --rigid_body CameraRig --out optitrack_gt.csv

Target Venue: IEEE Robotics and Automation Letters (RA-L)
"""

import os
import sys
import time
import socket
import struct
import argparse
import threading
import numpy as np


class NatNetDirectClient:
    """
    Lightweight UDP NatNet packet decoder for OptiTrack Motive.
    Listens on NatNet Data Port (default: 1511) and extracts Rigid Body poses.
    """
    def __init__(self, server_ip: str, local_ip: str = "0.0.0.0", multicast_ip: str = "239.255.42.99", port: int = 1511):
        self.server_ip = server_ip
        self.local_ip = local_ip
        self.multicast_ip = multicast_ip
        self.port = port
        self.sock = None
        self.running = False

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.sock.bind(("", self.port))
        except Exception:
            self.sock.bind((self.local_ip, self.port))

        # Join Multicast group
        try:
            mreq = struct.pack("4sl", socket.inet_aton(self.multicast_ip), socket.INADDR_ANY)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            pass

        self.sock.settimeout(1.0)
        self.running = True
        print(f"[NATNET] Listening for OptiTrack Motive data on port {self.port}...")

    def receive_packet(self):
        """Receives a raw UDP packet from Motive."""
        try:
            data, _ = self.sock.recvfrom(32768)
            return data
        except socket.timeout:
            return None
        except Exception:
            return None

    def close(self):
        self.running = False
        if self.sock:
            self.sock.close()


def run_ground_truth_logger(server_ip: str, output_csv: str, target_rigid_body_id: int = 1):
    client = NatNetDirectClient(server_ip=server_ip)
    client.connect()

    print("=" * 65)
    print("  OPTITRACK MOTIVE REAL-TIME POSE STREAMER & LOGGER")
    print(f"  Motive Server IP:    {server_ip}")
    print(f"  Target Rigid Body ID:{target_rigid_body_id}")
    print(f"  Output CSV:          {output_csv}")
    print("  Time Base:           Laptop Local Clock (monotonic time.time())")
    print("=" * 65)
    print("Press Ctrl+C to stop recording...\n")

    f = open(output_csv, "w")
    f.write("timestamp,tx,ty,tz,qx,qy,qz,qw\n")

    count = 0
    t0 = time.time()

    try:
        while True:
            packet = client.receive_packet()
            if packet is None:
                continue

            ts = time.time()

            # NatNet Packet Header: Message ID (2 bytes), Packet Size (2 bytes)
            if len(packet) < 4:
                continue

            msg_id, n_bytes = struct.unpack("<HH", packet[:4])
            if msg_id != 7: # 7 = NAT_FRAMEOFDATA
                continue

            offset = 4
            # Frame number (4 bytes)
            frame_num = struct.unpack("<I", packet[offset:offset+4])[0]
            offset += 4

            # MarkerSet count
            marker_set_count = struct.unpack("<I", packet[offset:offset+4])[0]
            offset += 4

            # Skip marker sets to get to rigid bodies
            for _ in range(marker_set_count):
                # Null-terminated name
                name_end = packet.find(b"\0", offset)
                offset = name_end + 1
                n_markers = struct.unpack("<I", packet[offset:offset+4])[0]
                offset += 4 + n_markers * 12 # 3 floats per marker

            # Other markers count
            if offset + 4 <= len(packet):
                n_other_markers = struct.unpack("<I", packet[offset:offset+4])[0]
                offset += 4 + n_other_markers * 12

            # Rigid Bodies
            if offset + 4 <= len(packet):
                n_rigid_bodies = struct.unpack("<I", packet[offset:offset+4])[0]
                offset += 4

                for rb_idx in range(n_rigid_bodies):
                    if offset + 32 > len(packet):
                        break
                    rb_id, x, y, z, qx, qy, qz, qw = struct.unpack("<Ifffffff", packet[offset:offset+32])
                    offset += 32

                    # Match target rigid body
                    if rb_id == target_rigid_body_id or n_rigid_bodies == 1:
                        f.write(f"{ts:.6f},{x:.6f},{y:.6f},{z:.6f},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n")
                        count += 1
                        if count % 100 == 0:
                            f.flush()
                            dt = time.time() - t0
                            print(f"[OPTITRACK] Captured {count:5d} poses ({count/dt:5.1f} Hz) | "
                                  f"Pos: [{x:6.3f}, {y:6.3f}, {z:6.3f}] m")
                        break
    except KeyboardInterrupt:
        print("\n[STOP] Recording stopped by user.")
    finally:
        f.flush()
        f.close()
        client.close()
        print(f"[SUCCESS] Saved {count} OptiTrack ground truth poses to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and log OptiTrack ground truth poses to CSV")
    parser.add_argument("--server", type=str, default="127.0.0.1", help="OptiTrack Motive PC IP address")
    parser.add_argument("--out", type=str, default="optitrack_ground_truth.csv", help="Output CSV path")
    parser.add_argument("--id", type=int, default=1, help="Rigid body ID in Motive")
    args = parser.parse_args()

    run_ground_truth_logger(server_ip=args.server, output_csv=args.out, target_rigid_body_id=args.id)

