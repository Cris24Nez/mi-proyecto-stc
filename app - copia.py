from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from flask_mysqldb import MySQL
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
app.secret_key = 'mi_clave_secreta'

# Configuración de la base de datos
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'miapp'

mysql = MySQL(app)


@app.route('/')
def index(): #pagina de inicio
    return render_template('index.html')  # página de inicio
#datos
@app.route('/iniciousuario', methods=['GET', 'POST'])  # Permitir GET y POST
def iniciousuario(): #sesión usuario
    if request.method == 'POST':
        usuario = request.form['usuario']
        contrasena = request.form['contrasena']

        cur = mysql.connection.cursor();
        cur.execute("SELECT * FROM usuarios WHERE usuario = %s AND contrasena = %s", (usuario, contrasena))
        cuenta = cur.fetchone()
        cur.close()

        if cuenta:
            session['usuario'] = usuario
            return redirect(url_for('opcion'))
        else:
            flash('Datos incorrectos')
            return redirect(url_for('iniciousuario')) 

    return render_template('iniciousuario.html')  # Mostrar formulario en GET

@app.route('/logout')
def logout():
    session.pop('usuario', None)  # Elimina el usuario de la sesión si existe
    return redirect(url_for('iniciousuario'))  # Redirige al login

@app.route('/opcion')
def opcion():
    if 'usuario' not in session:
        return redirect(url_for('iniciousuario'))
    # Aquí va el resto del código para mostrar la página de opciones
    return render_template('opcion.html')

#Datos paciente:
@app.route('/registropaciente', methods=['GET', 'POST'])
def registropaciente():
    if request.method == 'POST':
        ci = request.form['ci']
        nombre = request.form['nombre']
        apellido = request.form['apellido']
        edad = request.form['edad']
        genero = request.form['genero']
        telefono = request.form['telefono']
        ocupacion = request.form['ocupacion']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM pacientes WHERE ci = %s", (ci,))
        exist_paciente = cur.fetchone()
        
        if exist_paciente:
            flash("Paciente ya ingresado.", "warning")
            cur.close()
            return render_template('registropaciente.html', 
                                   ci=ci, nombre=nombre, apellido=apellido,
                                   edad=edad, genero=genero,
                                   telefono=telefono, ocupacion=ocupacion)
        
        # Insertar nuevo paciente
        cur.execute("""
            INSERT INTO pacientes (ci, nombre, apellido, edad, genero, telefono, ocupacion)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (ci, nombre, apellido, edad, genero, telefono, ocupacion))
        mysql.connection.commit()
        cur.close()
        
        flash("Paciente registrado exitosamente.", "success")
        # Limpiar los campos: no se pasan datos
        return render_template('registropaciente.html')

    return render_template('registropaciente.html')

@app.route('/buscarpaciente', methods=['GET', 'POST'])
def buscarpaciente():
    paciente = None
    if request.method == 'POST':
        ci = request.form['ci']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM pacientes WHERE ci = %s", (ci,))
        paciente = cur.fetchone()
        cur.close()
    if not paciente:
        flash("Paciente no encontrado", "danger")
    return render_template('buscarpaciente.html', paciente=paciente)

@app.route('/editarpaciente', methods=['GET', 'POST'])
def editarpaciente():
    paciente = None
    if request.method == 'POST':
        action = request.form.get('action')
        cur = mysql.connection.cursor()

        if action == 'buscar':
            ci = request.form['ci']
            cur.execute("SELECT * FROM pacientes WHERE ci = %s", (ci,))
            paciente = cur.fetchone()
            if not paciente:
                flash("Paciente no encontrado", "danger")

        elif action == 'editar':
            id = request.form['id']
            ci = request.form['ci']
            nombre = request.form['nombre']
            apellido = request.form['apellido']
            edad = request.form['edad']
            genero = request.form['genero']
            telefono = request.form['telefono']
            ocupacion = request.form['ocupacion']

            cur.execute("""
                UPDATE pacientes 
                SET ci=%s, nombre=%s, apellido=%s, edad=%s, genero=%s, telefono=%s, ocupacion=%s 
                WHERE id=%s
            """, (ci, nombre, apellido, edad, genero, telefono, ocupacion, id))
            mysql.connection.commit()
            flash("Datos actualizados exitosamente", "success")

            cur.execute("SELECT * FROM pacientes WHERE id = %s", (id,))
            paciente = cur.fetchone()

        cur.close()

    return render_template('editarpaciente.html', paciente=paciente)

@app.route('/eliminarpaciente', methods=['GET', 'POST'])
def eliminarpaciente():
    paciente = None

    if request.method == 'POST':
        action = request.form['action']

        if action == 'buscar':
            ci = request.form['ci']
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM pacientes WHERE ci = %s", (ci,))
            paciente = cursor.fetchone()
            cursor.close()

            if not paciente:
                flash('Paciente no encontrado.', 'danger')

        elif action == 'eliminar':
            paciente_id = request.form['id']
            cursor = mysql.connection.cursor()
            cursor.execute("DELETE FROM pacientes WHERE id = %s", (paciente_id,))
            mysql.connection.commit()
            cursor.close()
            flash('Paciente eliminado exitosamente.', 'success')
            return redirect(url_for('eliminarpaciente'))

    return render_template('eliminarpaciente.html', paciente=paciente)
#diagnostico
@app.route('/diagnostico')
def diagnostico():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, nombre FROM pacientes")
    pacientes = cursor.fetchall()
    cursor.close()
    return render_template('diagnostico.html', pacientes=pacientes)

@app.route('/guardar', methods=['POST'])
def guardar():
    if request.method == 'POST':
        id_paciente = request.form['id_paciente']
        fecha = request.form['fecha']
        mano_dominante = request.form['mano_dominante']
        gradosss = request.form['gradosss']
        gradofss = request.form['gradofss']
        observaciones = request.form['observaciones']
        
        cursor = mysql.connection.cursor()
        cursor.execute("""
            INSERT INTO diag1 (
                id_paciente, fecha, mano_dominante,
                gradosss, gradofss, observaciones
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (id_paciente, fecha, mano_dominante,
              gradosss, gradofss, observaciones))
        mysql.connection.commit()
        cursor.close()
        return redirect('/opcion')
    
#diagnóstico práctico
@app.route('/diagnosticop1')
def diagnosticop1(): #pagina de inicio
    return render_template('diagnosticop1.html')

@app.route('/diagnosticop2')
def diagnosticop2(): #pagina de inicio
    return render_template('diagnosticop2.html')

@app.route('/diagnosticop3')
def diagnosticop3(): #pagina de inicio
    return render_template('diagnosticop3.html')

#rutina de ejercicios pasivos/activos
@app.route('/ejerpasivos')
def ejerpasivos(): #Ejercicios pasivos 
    return render_template('ejerpasivos.html')

@app.route('/ejerpactivos')
def ejeractivos(): #Ejercicios activos
    return render_template('ejeractivos.html')

#Ejercicios activos con mediapipe
# Variable global para detener el ejercicio
ejercicio_activo = True

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

#Ejercicio 1: Abrir y cerrar la mano
def gen_frames():
    prev_state = None
    open_count = 0
    close_count = 0
    exercise_started = False
    exercise_ended = False
    start_time = None  # No se define hasta detectar la mano
    exercise_duration = 30  # duración del ejercicio en segundos

    global ejercicio_activo
    while ejercicio_activo:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        white_bg = 255 * np.ones_like(frame)

        if results.multi_hand_landmarks and not exercise_ended:
            if not exercise_started:
                exercise_started = True
                start_time = time.time()  # El tiempo comienza cuando se detecta la mano

            elapsed_time = current_time - start_time
            remaining = max(0, int(exercise_duration - elapsed_time))
            if elapsed_time >= exercise_duration:
                exercise_ended = True
        elif exercise_started and not exercise_ended:
            elapsed_time = current_time - start_time
            remaining = max(0, int(exercise_duration - elapsed_time))
            if elapsed_time >= exercise_duration:
                exercise_ended = True
        else:
            remaining = exercise_duration  # Mostrar tiempo completo hasta que comience

        # Dibujo y conteo
        if results.multi_hand_landmarks and not exercise_ended:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(white_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                fingers = [8, 12, 16, 20]
                extended = sum(1 for tip in fingers if landmarks[tip].y < landmarks[tip - 2].y)
                if landmarks[4].x > landmarks[3].x:
                    extended += 1

                current_state = "abierta" if extended >= 4 else "cerrada"
                color = (0, 255, 0) if current_state == "abierta" else (0, 0, 255)

                if prev_state != current_state and prev_state is not None:
                    if current_state == "abierta":
                        open_count += 1
                    else:
                        close_count += 1

                prev_state = current_state

                cv2.putText(white_bg, f"Estado: Mano {current_state}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Mostrar siempre el temporizador y conteos
        timer_color = (0, 0, 255) if remaining <= 5 else (100, 0, 100)
        cv2.putText(white_bg, f"Tiempo restante: {remaining}s", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2)
        cv2.putText(white_bg, f"Aperturas: {open_count}", (400, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 50, 0), 2)
        cv2.putText(white_bg, f"Cierres: {close_count}", (400, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 150), 2)

        if exercise_ended:
            cv2.putText(white_bg, "Ejercicio concluido", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', white_bg)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()
    
@app.route('/index0')
def index0(): #abrir y cerrar la mano
    return render_template('index0.html')

#Ejercicio 2: Coordinacion
def gen_frames():
    global ejercicio_activo
    # Datos del ejercicio
    finger_names = ["Indice", "Medio", "Anular", "Menique"]
    finger_tips = [8, 12, 16, 20]
    current_finger = 0
    touch_detected = False
    touch_counter = 0

    # Timer
    start_time = None
    exercise_duration = 40  # segundos
    exercise_done = False

    # Colores
    colors = [(255, 255, 255), (200, 255, 200)]  # Normal, Éxito

    while cap.isOpened() and ejercicio_activo:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        white_bg = np.ones_like(frame) * 255  # fondo blanco

        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    white_bg,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Obtener puntos clave
                lm = hand_landmarks.landmark
                thumb_tip = np.array([int(lm[4].x * width), int(lm[4].y * height)])
                finger_tip = np.array([int(lm[finger_tips[current_finger]].x * width),
                                       int(lm[finger_tips[current_finger]].y * height)])

                # Escala de referencia para umbral
                base_thumb = np.array([int(lm[1].x * width), int(lm[1].y * height)])
                base_index = np.array([int(lm[5].x * width), int(lm[5].y * height)])
                scale_ref = np.linalg.norm(base_thumb - base_index)
                threshold = scale_ref * 0.4

                # Calcular distancia real
                distance = np.linalg.norm(thumb_tip - finger_tip)

                if distance < threshold:
                    if not touch_detected:
                        touch_counter += 1
                        touch_detected = True
                        current_finger += 1
                        if current_finger >= len(finger_tips):
                            current_finger = 0
                else:
                    touch_detected = False

                # Visualizar estado
                cv2.putText(white_bg, f"Tocar con: {finger_names[current_finger]}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.circle(white_bg, tuple(thumb_tip), 10, (0, 0, 200), -1)
                cv2.circle(white_bg, tuple(finger_tip), 10, (0, 200, 0), -1)

        if start_time:
            elapsed_time = time.time() - start_time
            if elapsed_time >= exercise_duration:
                exercise_done = True
                cv2.putText(white_bg, "Ejercicio concluido", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                remaining = int(exercise_duration - elapsed_time)
                cv2.putText(white_bg, f"Tiempo restante: {remaining}s", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 0, 100), 2)

        # Mostrar contador de aciertos
        cv2.putText(white_bg, f"Toques correctos: {touch_counter}", (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 0), 2)

        # Enviar el frame codificado al navegador
        ret, buffer = cv2.imencode('.jpg', white_bg)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Al salir del bucle se liberan los recursos
    cap.release()
    cv2.destroyAllWindows()

@app.route('/index1')
def index1(): #pagina de inicio
    return render_template('index1.html')

#Ejercicio 3 de levantamiento de dedos
def gen_frames():
    global ejercicio_activo
    finger_tips = [20, 16, 12, 8, 4]
    finger_names = ["Menique", "Anular", "Medio", "Indice", "Pulgar"]
    current_finger_index = 0
    direction = "up"
    raised_fingers = set()
    cycles_completed = 0

    start_time = None
    exercise_duration = 60
    exercise_done = False

    message = ""
    message_color = (0, 0, 0)
    last_action_time = 0
    action_interval = 0.4

    while cap.isOpened() and ejercicio_activo:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        white_bg = 255 * np.ones_like(frame)

        if results.multi_hand_landmarks and not exercise_done:
            if start_time is None:
                start_time = time.time()

            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            mp_drawing.draw_landmarks(
                white_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = hand_landmarks.landmark

            if time.time() - last_action_time > action_interval:
                if current_finger_index < len(finger_tips):
                    tip = finger_tips[current_finger_index]
                    pip = tip - 2

                    if direction == "up":
                        if tip == 4:  # Pulgar
                            if (handedness == "Right" and landmarks[tip].x < landmarks[pip].x) or \
                               (handedness == "Left" and landmarks[tip].x > landmarks[pip].x):
                                raised_fingers.add(current_finger_index)
                                message = f"{finger_names[current_finger_index]} levantado"
                                message_color = (0, 120, 0)
                                current_finger_index += 1
                                last_action_time = time.time()
                        else:
                            if landmarks[tip].y < landmarks[pip].y:
                                raised_fingers.add(current_finger_index)
                                message = f"{finger_names[current_finger_index]} levantado"
                                message_color = (0, 120, 0)
                                current_finger_index += 1
                                last_action_time = time.time()

                        if current_finger_index == len(finger_tips):
                            direction = "down"
                            current_finger_index = 0
                            message = "¡Todos los dedos levantados! Comienza el descenso"
                            message_color = (150, 0, 0)
                            last_action_time = time.time()

                    elif direction == "down":
                        if tip == 4:
                            if (handedness == "Right" and landmarks[tip].x > landmarks[pip].x) or \
                               (handedness == "Left" and landmarks[tip].x < landmarks[pip].x):
                                raised_fingers.discard(current_finger_index)
                                message = f"{finger_names[current_finger_index]} bajado"
                                message_color = (0, 0, 150)
                                current_finger_index += 1
                                last_action_time = time.time()
                        else:
                            if landmarks[tip].y > landmarks[pip].y:
                                raised_fingers.discard(current_finger_index)
                                message = f"{finger_names[current_finger_index]} bajado"
                                message_color = (0, 0, 150)
                                current_finger_index += 1
                                last_action_time = time.time()

                        if current_finger_index == len(finger_tips):
                            direction = "up"
                            current_finger_index = 0
                            cycles_completed += 1
                            message = f"Ciclo {cycles_completed} completado"
                            message_color = (0, 0, 255)
                            last_action_time = time.time()

        if start_time:
            elapsed = int(time.time() - start_time)
            if elapsed >= exercise_duration:
                exercise_done = True
            remaining = max(0, exercise_duration - elapsed)
            cv2.putText(white_bg, f"Tiempo restante: {remaining}s", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)
            bar_length = int(400 * (remaining / exercise_duration))
            cv2.rectangle(white_bg, (10, 490), (10 + bar_length, 510), (0, 255, 0), -1)

        if message:
            cv2.putText(white_bg, message, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, message_color, 2)

        if not exercise_done:
            action = "Levantar" if direction == "up" else "Bajar"
            instruction = f"{action} {finger_names[current_finger_index]}" if current_finger_index < len(finger_names) else ""
            cv2.putText(white_bg, instruction, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(white_bg, f"Ciclos completados: {cycles_completed}", (400, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 50), 2)

        if exercise_done:
            cv2.putText(white_bg, "¡Ejercicio finalizado!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', white_bg)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/index2')
def index2(): #pagina de inicio
    return render_template('index2.html')

#Ejercicio 4 de flexion-extension
target_fingers = ["Indice", "Medio", "Anular", "Menique"]
finger_joints = {
    "Indice": [5, 6, 7, 8],
    "Medio": [9, 10, 11, 12],
    "Anular": [13, 14, 15, 16],
    "Menique": [17, 18, 19, 20]
}
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def gen_frames():
    global ejercicio_activo
    direction = "down"  # Empezamos con flexión
    flexion_count = 0
    extension_count = 0
    message = ""
    message_color = (0, 0, 0)
    #timer
    start_time = None
    exercise_duration = 60
    exercise_done = False

    # Para evitar conteos múltiples rápidos
    last_change_time = 0
    min_interval = 0.7  # segundos

    while cap.isOpened() and ejercicio_activo:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        white_bg = 255 * np.ones_like(frame)

        if results.multi_hand_landmarks and not exercise_done:
            if start_time is None:
                start_time = time.time()

            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                white_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            landmarks = hand_landmarks.landmark

            all_fingers_done = True
            
            for finger in target_fingers:
                mcp, pip, dip, tip = finger_joints[finger]
                angle = calculate_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
                
                if direction == "down":  # Queremos flexión: ángulo < 120°
                    if angle >= 120:
                        all_fingers_done = False
                else:  # direction == "up", queremos extensión: ángulo > 150°
                    if angle <= 150:
                        all_fingers_done = False

            current_time = time.time()
            if all_fingers_done and (current_time - last_change_time > min_interval):
                if direction == "down":
                    message = "Flexionados los 4 dedos"
                    message_color = (0, 0, 150)
                    flexion_count += 1
                    direction = "up"
                else:
                    message = "Extendidos los 4 dedos"
                    message_color = (0, 150, 0)
                    extension_count += 1
                    direction = "down"
                last_change_time = current_time

        if start_time:
            elapsed = int(time.time() - start_time)
            if elapsed >= exercise_duration:
                exercise_done = True
            remaining = max(0, exercise_duration - elapsed)
            cv2.putText(white_bg, f"Tiempo restante: {remaining}s", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

        if message:
            cv2.putText(white_bg, message, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 3)

        if not exercise_done:
            accion = "Flexionar" if direction == "down" else "Extender"
            instruccion = f"{accion}"
            cv2.putText(white_bg, instruccion, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Mostrar contadores en la parte inferior (abajo)
            cv2.putText(white_bg, f"Flexiones: {flexion_count}", (10, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 50), 2)
            cv2.putText(white_bg, f"Extensiones: {extension_count}", (250, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 50), 2)

        if exercise_done:
            cv2.putText(white_bg, "Ejercicio finalizado", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

         # Enviar el frame codificado al navegador
        ret, buffer = cv2.imencode('.jpg', white_bg)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    
@app.route('/index3')
def index3(): #pagina de inicio
    return render_template('index3.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detener', methods=['POST'])
def detener():
    global ejercicio_activo
    ejercicio_activo = False
    return jsonify({'mensaje': 'Ejercicio detenido correctamente'})

@app.route('/reiniciar', methods=['POST'])
def reiniciar():
    global ejercicio_activo, cap, hands
    # Reiniciamos las variables de control
    ejercicio_activo = True
    # Reabrir la cámara
    cap.release()
    cap = cv2.VideoCapture(0)
    # Reiniciar el detector de manos si se requiere
    hands.close()  # Liberar el recurso anterior
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return jsonify({'mensaje': 'Ejercicio reiniciado correctamente'})

#ejercicio de rehabilitacion
@app.route('/progresopaciente')
def progresopaciente(): #pagina de inicio
    return render_template('progresopaciente.html')

#inicioPaciente
@app.route('/iniciopaciente', methods=['GET', 'POST'])
def iniciopaciente():
    paciente = None
    if request.method == 'POST':
        ci = request.form['ci']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM pacientes WHERE ci = %s", (ci,))
        paciente = cur.fetchone()
        cur.close()
    if not paciente:
        flash("Paciente no encontrado", "danger")
    return render_template('iniciopaciente.html', paciente=paciente)

#inicioAdmin
@app.route('/inicioadmin', methods=['GET', 'POST'])  # Permitir GET y POST
def inicioadmin():
    if request.method == 'POST':
        usuario = request.form['usuario']
        contrasena = request.form['contrasena']

        # Validación simple
        if usuario == 'admin' and contrasena == '1234':
            return redirect(url_for('servicio'))  # Ruta protegida
        else:
            flash("Usuario o contraseña incorrectos")
            return redirect(url_for('inicioadmin'))  # Volver al login si falla
        
    return render_template('inicioadmin.html')  # Mostrar formulario en GET

@app.route('/servicio')
def servicio(): #pagina de servicio tecnico
    return render_template('servicio.html')

@app.route('/registrousuario', methods=['GET', 'POST'])
def registrousuario():
    if request.method == 'POST':
        usuario = request.form['usuario']
        contrasena = request.form['contrasena']

        cur = mysql.connection.cursor()

        # Verificar si el usuario ya existe
        cur.execute("SELECT * FROM usuarios WHERE usuario = %s", (usuario,))
        exist_usuario = cur.fetchone()

        if exist_usuario:
            flash('Usuario existente: {}'.format(usuario), 'warning')
            cur.close()
            # Reenviar los datos al formulario
            return render_template('registrousuario.html', usuario=usuario)

        # Si no existe, registrar
        cur.execute("INSERT INTO usuarios (usuario, contrasena) VALUES (%s, %s)", (usuario, contrasena))
        mysql.connection.commit()
        cur.close()

        flash('Usuario registrado exitosamente', 'success')
        return redirect(url_for('registrousuario'))  # Recarga el formulario vacío

    return render_template('registrousuario.html')

@app.route('/historial_mantenimiento')
def historial_mantenimiento():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM mantenimiento ORDER BY fecha DESC")
    mantenimientos = cur.fetchall()
    cur.close()
    return render_template('historial_mantenimiento.html', mantenimientos=mantenimientos)

@app.route('/nuevo_mantenimiento', methods=['GET', 'POST'])
def nuevo_mantenimiento():
    if request.method == 'POST':
        fecha = request.form['fecha']
        descripcion = request.form['descripcion']
        observaciones = request.form['observaciones']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO mantenimiento (fecha, descripcion, observaciones) VALUES (%s, %s, %s)",
                    (fecha, descripcion, observaciones))
        mysql.connection.commit()
        cur.close()

        flash('Mantenimiento registrado exitosamente', 'success')
        return redirect(url_for('historial_mantenimiento'))  # Redirige al historial

    return render_template('nuevo_mantenimiento.html')  # Muestra el formulario

if __name__ == '__main__':
    app.run(debug=True)
