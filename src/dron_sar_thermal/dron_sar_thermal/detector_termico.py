#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from rclpy.qos import qos_profile_sensor_data
import math

def on_trackbar(val):
    pass 

class DetectorTermico(Node):
    def __init__(self):
        super().__init__('detector_termico')
        
        self.subscription = self.create_subscription(
            Image,
            '/dron/thermal_camera/image_raw', 
            self.image_callback,
            qos_profile_sensor_data)
            
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            qos_profile_sensor_data)
            
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_callback,
            qos_profile_sensor_data)
            
        self.vel_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.br = CvBridge()
        
        # Constantes de control
        self.centro_pantalla_x = 320
        self.centro_pantalla_y = 240
        self.Kp_avance = 0.004   
        self.Kp_giro = 0.0035     
        self.zona_muerta = 30    
        self.vel_max = 1.0       
        self.SIGNO_AVANCE = -1  
        self.SIGNO_GIRO   = -1   

        # Variables de estado
        self.mision_interrumpida = False
        self.ultima_direccion_giro = 0.0
        self.tiempo_perdida_objetivo = 0.0
        self.TIEMPO_MAXIMO_RADAR = 15.0 
        self.latitud = 0.0
        self.longitud = 0.0
        self.altitud = 0.0
        self.reporte_enviado = False 

        # --- MÁQUINA DE ESTADOS MULTI-OBJETIVO ---
        self.estado_maniobra = "VISUAL"  
        self.inicio_maniobra = 0.0
        self.duracion_maniobra = 0.0
        self.vel_giro_ciego = 0.0

        # Definición de la Cuadrícula (Zona Roja y Verde)
        self.limite_izq = int(640 * 0.2)   
        self.limite_der = int(640 * 0.8)   
        self.limite_sup = int(480 * 0.2)   
        self.limite_inf = int(480 * 0.8)   

        cv2.namedWindow("Mascara de Deteccion Infrarroja")
        cv2.createTrackbar("Temp_Min", "Mascara de Deteccion Infrarroja", 0, 255, on_trackbar)
        cv2.createTrackbar("Temp_Max", "Mascara de Deteccion Infrarroja", 1, 255, on_trackbar)

        self.get_logger().info("Nodo SAR inicializado. IA lista para patrullaje continuo...")

    def gps_callback(self, msg):
        self.latitud = msg.latitude
        self.longitud = msg.longitude
        self.altitud = msg.altitude

    def generar_reporte_rescate(self, frame_captura):
        timestamp = int(time.time())
        nombre_archivo = f"evidencia_rescate_{timestamp}.jpg"
        cv2.imwrite(nombre_archivo, frame_captura)
        
        mensaje_alerta = f"""
        ====================================================
        [ TRANSMISIÓN RF LoRa SIMULADA ]
        ¡ VÍCTIMA LOCALIZADA CON ÉXITO !
        
        -> LATITUD:   {self.latitud:.7f}
        -> LONGITUD:  {self.longitud:.7f}
        -> ALTITUD:   {self.altitud:.2f} metros
        
        [!] Evidencia visual '{nombre_archivo}' guardada.
        ====================================================
        """
        self.get_logger().error(mensaje_alerta)

    def state_callback(self, msg):
        # Evitamos reiniciar las variables si estamos en el Cooldown de escape
        if msg.mode == "AUTO" and self.mision_interrumpida and self.estado_maniobra == "VISUAL":
            self.get_logger().info("Modo AUTO forzado externamente. Reiniciando IA...")
            self.mision_interrumpida = False
            self.reporte_enviado = False 

    def cambiar_modo(self, modo_deseado):
        if self.set_mode_client.wait_for_service(timeout_sec=1.0):
            req = SetMode.Request()
            req.custom_mode = modo_deseado
            self.set_mode_client.call_async(req)
            if modo_deseado == "GUIDED":
                self.get_logger().warn("¡FIRMA TÉRMICA DETECTADA! Abandonando ruta AUTO...")
            elif modo_deseado == "AUTO":
                self.get_logger().info("Retomando ruta de patrullaje AUTO...")
        else:
            self.get_logger().error("Error de servicio MAVROS.")

    def image_callback(self, data):
        frame = self.br.imgmsg_to_cv2(data, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        t_min = cv2.getTrackbarPos("Temp_Min", "Mascara de Deteccion Infrarroja")
        t_max = cv2.getTrackbarPos("Temp_Max", "Mascara de Deteccion Infrarroja")
        mask = cv2.inRange(gray, t_min, t_max)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        msg_target = PositionTarget()
        msg_target.coordinate_frame = 8
        msg_target.type_mask = 1 | 2 | 4 | 64 | 128 | 256 | 1024
        msg_target.velocity.y = 0.0
        msg_target.velocity.z = 0.0

        tiempo_actual = time.time()

        # ========================================================
        # MÁQUINA DE ESTADOS: SECUENCIAS AUTOMATIZADAS
        # ========================================================
        if self.estado_maniobra != "VISUAL":
            tiempo_transcurrido = tiempo_actual - self.inicio_maniobra
            
            # FASE 1: Giro Ciego
            if self.estado_maniobra == "GIRO_CIEGO":
                if tiempo_transcurrido < self.duracion_maniobra:
                    msg_target.yaw_rate = self.vel_giro_ciego
                    msg_target.velocity.x = 0.0
                    self.vel_pub.publish(msg_target)
                else:
                    self.estado_maniobra = "PAUSA_CIEGA"
                    self.inicio_maniobra = tiempo_actual
                    self.duracion_maniobra = 2.0  
            
            # FASE 2: Freno en seco
            elif self.estado_maniobra == "PAUSA_CIEGA":
                if tiempo_transcurrido < self.duracion_maniobra:
                    msg_target.yaw_rate = 0.0
                    msg_target.velocity.x = 0.0
                    self.vel_pub.publish(msg_target)
                else:
                    self.estado_maniobra = "AVANCE_CIEGO"
                    self.inicio_maniobra = tiempo_actual
                    self.duracion_maniobra = 3.5  
                    
            # FASE 3: Avance Ciego hacia el objetivo
            elif self.estado_maniobra == "AVANCE_CIEGO":
                if tiempo_transcurrido < self.duracion_maniobra:
                    msg_target.yaw_rate = 0.0
                    msg_target.velocity.x = 0.8  # Avance puro hacia adelante
                    self.vel_pub.publish(msg_target)
                else:
                    self.estado_maniobra = "VISUAL"
                    
            # FASE 4: Flotar sobre la víctima luego de tomar la foto (5 SEGUNDOS)
            elif self.estado_maniobra == "FLOTAR_RESCATE":
                if tiempo_transcurrido < self.duracion_maniobra:
                    msg_target.yaw_rate = 0.0
                    msg_target.velocity.x = 0.0
                    self.vel_pub.publish(msg_target)
                else:
                    self.get_logger().info("Tiempo de reporte finalizado. Regresando a misión...")
                    self.cambiar_modo("AUTO")
                    
                    self.estado_maniobra = "COOLDOWN_ESCAPE"
                    self.inicio_maniobra = tiempo_actual
                    self.duracion_maniobra = 10.0 # 10 segundos de ceguera para alejarse
            
            # FASE 5: Cooldown de escape (Ignoramos la cámara mientras ArduPilot vuela)
            elif self.estado_maniobra == "COOLDOWN_ESCAPE":
                if tiempo_transcurrido >= self.duracion_maniobra:
                    self.get_logger().info("Zona de rescate abandonada. IA visual reactivada.")
                    self.estado_maniobra = "VISUAL"
                    self.mision_interrumpida = False
                    self.reporte_enviado = False

            self.dibujar_cuadricula(frame)
            cv2.imshow("Camara Termica", frame)
            cv2.imshow("Mascara de Deteccion Infrarroja", mask)
            cv2.waitKey(1)
            return 

        # ========================================================
        # MODO VISUAL NORMAL (ZONA VERDE Y DETECCIÓN)
        # ========================================================
        objetivo_detectado = False

        if contours:
            contorno_principal = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contorno_principal)
            
            if area > 80: 
                objetivo_detectado = True
                self.tiempo_perdida_objetivo = tiempo_actual 
                
                x, y, w, h = cv2.boundingRect(contorno_principal)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cx, cy = x + w//2, y + h//2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                if not self.mision_interrumpida:
                    self.cambiar_modo("GUIDED")
                    self.mision_interrumpida = True

                error_x = cx - self.centro_pantalla_x
                error_y = cy - self.centro_pantalla_y

                en_zona_roja = (cx < self.limite_izq) or (cx > self.limite_der) or (cy < self.limite_sup) or (cy > self.limite_inf)

                if en_zona_roja:
                    grados_error = (error_x / 320.0) * 40.0
                    self.get_logger().warn(f"Objetivo en Zona ROJA detectado a {abs(grados_error):.1f}°. Iniciando maniobra ciega...")
                    
                    velocidad_giro = 0.4 
                    angulo_rad = abs(grados_error) * (math.pi / 180.0)
                    
                    self.duracion_maniobra = angulo_rad / velocidad_giro
                    direccion = 1 if error_x > 0 else -1
                    self.vel_giro_ciego = velocidad_giro * direccion * self.SIGNO_GIRO
                    
                    self.estado_maniobra = "GIRO_CIEGO"
                    self.inicio_maniobra = tiempo_actual
                    
                    msg_target.yaw_rate = 0.0
                    msg_target.velocity.x = 0.0
                    self.vel_pub.publish(msg_target)
                    
                else:
                    if not self.reporte_enviado:
                        if abs(error_x) < 80 and abs(error_y) < 80:
                            self.get_logger().warn("¡Objetivo centrado! Disparando captura...")
                            self.generar_reporte_rescate(frame)
                            self.reporte_enviado = True 
                            
                            # --- INICIAMOS EL FLOTADO POST-RESCATE ---
                            self.get_logger().info("Flotando 5 segundos para transmisión de datos...")
                            self.estado_maniobra = "FLOTAR_RESCATE"
                            self.inicio_maniobra = tiempo_actual
                            self.duracion_maniobra = 5.0
                            return 
                    
                    if abs(error_x) < 20: 
                        giro_crudo = 0.0 
                    else:
                        limite_giro = 0.25 
                        giro_calculado = float(error_x * self.Kp_giro * self.SIGNO_GIRO)
                        giro_crudo = max(min(giro_calculado, limite_giro), -limite_giro)

                    if abs(error_y) < self.zona_muerta:
                        avance_crudo = 0.0 
                    else:
                        avance_crudo = float(error_y * self.Kp_avance * self.SIGNO_AVANCE) 
                    
                    self.ultima_direccion_giro = giro_crudo
                    msg_target.yaw_rate = max(min(giro_crudo, self.vel_max), -self.vel_max)
                    msg_target.velocity.x = max(min(avance_crudo, self.vel_max), -self.vel_max)
                    self.vel_pub.publish(msg_target)

        # Radar de pérdida de señal
        if not objetivo_detectado and self.mision_interrumpida and self.estado_maniobra == "VISUAL":
            if tiempo_actual - self.tiempo_perdida_objetivo > self.TIEMPO_MAXIMO_RADAR:
                self.cambiar_modo("AUTO")
                self.mision_interrumpida = False
                msg_target.velocity.x = 0.0
                msg_target.yaw_rate = 0.0
            else:
                msg_target.velocity.x = 0.0
                msg_target.yaw_rate = 0.4 if self.ultima_direccion_giro >= 0 else -0.4
            self.vel_pub.publish(msg_target)

        self.dibujar_cuadricula(frame)
        cv2.imshow("Camara Termica", frame)
        cv2.imshow("Mascara de Deteccion Infrarroja", mask)
        cv2.waitKey(1)

    def dibujar_cuadricula(self, frame):
        cv2.rectangle(frame, (self.limite_izq, self.limite_sup), (self.limite_der, self.limite_inf), (0, 200, 0), 2)
        cv2.putText(frame, f"ESTADO: {self.estado_maniobra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def main(args=None):
    rclpy.init(args=args)
    detector = DetectorTermico()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()