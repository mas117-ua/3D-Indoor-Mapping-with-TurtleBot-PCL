#include <memory>
#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex> 

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

//para filtras nans que se producen con bordes antes de voxel grid
#include <pcl/filters/filter.h>

//normales
#include <pcl/features/normal_3d.h>

//filtrar normales no validas
#include <pcl/common/io.h>

//keypoints
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

//descriptor
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>

//matching
#include <pcl/registration/correspondence_estimation.h>

//ransac
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <Eigen/Core>

#include <pcl/common/transforms.h>

#include <pcl/registration/icp.h>


bool USAR_ISS = true;
bool USAR_FPFH = true;



float VOXEL_GRID_SIZE = 0.02f;
float NORMAL_RADIUS = 0.08f;

float FPFHS_RADIUS = 0.15f;
float PFH_RADIUS = 0.15f;

// =========================================================
// PARÁMETROS CONFIGURABLES 
// =========================================================

// Visualizador
const int VISU_SLEEP_MS = 100;               // Tiempo de espera (ms) para el refresco del CloudViewer

// Subscriptor ROS
const std::string TOPIC_NAME = "/camera/depth/points"; // Nombre del topic de la cámara de profundidad
const int QUEUE_SIZE = 10;                   // Tamaño de la cola de mensajes del subscriptor

// Parámetros del detector ISS3D
const double ISS_GAMMA_21 = 0.85;            // Umbral de relación entre el 2º y 1º autovalor (Non-max suppression)
const double ISS_GAMMA_32 = 0.85;            // Umbral de relación entre el 3º y 2º autovalor
const int ISS_THREADS = 4;  
// Define hasta qué distancia miramos alrededor de un punto para decidir si tiene forma de esquina.                 
float ISS_SALIENT_RADIUS = 0.06f;
//Si el algoritmo encuentra 10 puntos clave buenísimos amontonados en la misma esquina de la mesa, borra los 9 peores y 
    // se queda solo con el más fuerte (el centro exacto de la esquina).
float ISS_NON_MAX_RADIUS = 0.025f;

// Parámetros del detector Harris3D
const float HARRIS_THRESHOLD =  0.00001f;//0.0001f; //0.001f;       // Umbral de respuesta de esquina para ser considerada válida
float HARRIS_SALIENT_RADIUS = 0.15f; //0.20f;// debe ser mucho mas mayor que el voxel grid       //0.06f;         // Escala principal de vecindad para Harris3D
float HARRIS_NON_MAX_RADIUS = 0.02f;//0.025f;        // Junto con SALIENT alimenta el único setRadius() de PCL Harris (normales+NMS)

// Parámetros de RANSAC y Correspondencias
const size_t MIN_CORRESPONDENCES_RANSAC = 3;    //4; // Correspondencias mínimas necesarias para ejecutar RANSAC
const double RANSAC_INLIER_DIST = 0.08f;       // Distancia máxima (m) para que un par coincidente sea considerado inlier
const int RANSAC_MAX_ITER = 1000;            // Número máximo de iteraciones permitidas en el algoritmo RANSAC
const size_t MIN_INLIERS_TRANSFORM = 3; //8 y iba bien en opcion 1; //3;      // Inliers mínimos para aceptar como válida una transformación rígida

// Parámetros del filtro de cordura (Sanity Check de movimiento)
const float MAX_TRANSLATION_JUMP = 0.30f;    // Salto de traslación máximo permitido en metros (dx, dy, dz)
const float MAX_ROTATION_JUMP = 15.0f;//45.0f;       // Giro máximo permitido en grados (eje Yaw)

// Variables globales de estado del pipeline
pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_previos(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptores_previos(new pcl::PointCloud<pcl::FPFHSignature33>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapa_global(new pcl::PointCloud<pcl::PointXYZRGB>());
Eigen::Matrix4f transformacion_global = Eigen::Matrix4f::Identity();

// Para guardar los descriptores previos si usamos PFH
pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptores_previos_pfh(new pcl::PointCloud<pcl::PFHSignature125>);



// Memoria para ICP
pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_previa(new pcl::PointCloud<pcl::PointXYZRGB>);

// -----------------------------------------------------------------------
// FIX 1: Mutex para proteger el acceso concurrente a visu_pc entre el
// hilo del visualizador (simpleVis) y el callback de ROS2.
// Sin esto, el visualizador puede leer el puntero mientras el callback
// lo está reemplazando -> segmentation fault.
// -----------------------------------------------------------------------
pcl::PointCloud<pcl::PointXYZRGB>::Ptr visu_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
std::mutex visu_mutex;
bool viewer_started = false;

void simpleVis() {
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer_started = true;
    while (!viewer.wasStopped()) {
        // FIX 1b: Copiamos el puntero bajo el mutex para no bloquearlo
        // durante el renderizado (que puede tardar varios ms).
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_local;
        {
            std::lock_guard<std::mutex> lock(visu_mutex);
            nube_local = visu_pc;
        }
        if (nube_local && nube_local->size() > 0) {
            viewer.showCloud(nube_local);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(VISU_SLEEP_MS));
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtrado(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_filtrada(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<int> indices_puntos_validos;
    pcl::removeNaNFromPointCloud(*cloud, *nube_filtrada, indices_puntos_validos);
    std::cout << "Puntos filtrados: " << nube_filtrada->size() << std::endl;
    return nube_filtrada;
}

pcl::PointCloud<pcl::Normal>::Ptr calcularNormales(pcl::PointCloud<pcl::PointXYZRGB>::Ptr nubes_pos_vg) {
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> estimador;
    estimador.setInputCloud(nubes_pos_vg);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr arbol_busqueda(new pcl::search::KdTree<pcl::PointXYZRGB>());
    estimador.setSearchMethod(arbol_busqueda);
    estimador.setRadiusSearch(NORMAL_RADIUS);
    pcl::PointCloud<pcl::Normal>::Ptr normales(new pcl::PointCloud<pcl::Normal>);
    estimador.compute(*normales);
    return normales;
}

// ---------------------------------------------------------
// KEYPOINTS ISS3D
// ---------------------------------------------------------


pcl::PointCloud<pcl::PointXYZRGB>::Ptr extraerKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_entrada) {
    pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;
    // Ayuda al ordenador a encontrar qué puntos están cerca de otros de forma casi instantánea, 
    // sin tener que buscar por todo el mapa punto por punto.
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    iss_detector.setSearchMethod(tree);
    // Define hasta qué distancia miramos alrededor de un punto para decidir si tiene forma de esquina.
    iss_detector.setSalientRadius(ISS_SALIENT_RADIUS);
    //Si el algoritmo encuentra 10 puntos 
    // clave buenísimos amontonados en la misma esquina de la mesa, borra los 9 peores y 
    // se queda solo con el más fuerte (el centro exacto de la esquina).
    iss_detector.setNonMaxRadius(ISS_NON_MAX_RADIUS);
    // Define cuánto debe cambiar la forma a mi alrededor para que se considere una esquina.
    iss_detector.setThreshold21(ISS_GAMMA_21);
    // Define cuánto debe cambiar la forma a mi alrededor para que se considere una esquina.
    iss_detector.setThreshold32(ISS_GAMMA_32);
    iss_detector.setNumberOfThreads(ISS_THREADS);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);
    iss_detector.setInputCloud(cloud_entrada);
    iss_detector.compute(*keypoints);
    return keypoints;
}

// ---------------------------------------------------------
// DESCRIPTORES FPFH
// ---------------------------------------------------------

pcl::PointCloud<pcl::FPFHSignature33>::Ptr extraerDescriptores(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_entrada,
    pcl::PointCloud<pcl::Normal>::Ptr normales_finales,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_final) {
    //DE TAMAÑO 33,3 HISTOGRAMAS DE 11 BINS
    pcl::FPFHEstimation<pcl::PointXYZRGB ,pcl::Normal , pcl::FPFHSignature33>fpfh_estimador;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    fpfh_estimador.setSearchMethod(tree);
    // Le pasamos los puntos clave a los que queremos hacerles el "carnet".
    fpfh_estimador.setInputCloud(keypoints_entrada);
    fpfh_estimador.setSearchSurface(nube_final);
    // FPFH depende totalmente de las normales (la orientación de la geometría) para calcular los ángulos.
    fpfh_estimador.setInputNormals(normales_finales);
    fpfh_estimador.setRadiusSearch(FPFHS_RADIUS);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptores(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh_estimador.compute(*descriptores);

    // SINCRONIZA KEYPOINTS Y DESCRIPTORES VALIDOS
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc_validos(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_validos(new pcl::PointCloud<pcl::PointXYZRGB>);

    //FPFH puede dejar descriptores inválidos cuando, por ejemplo, hay pocos vecinos en el radio de búsqueda, 
    // normales mal estimadas o puntos degenerados. Esos casos suelen propagarse a todo el histograma; 
    // comprobar histogram[0] es un criterio rápido para detectar descriptores “rotos” sin validar los 33 valores uno a uno
    //  (en la práctica, si el descriptor falla, muchas veces varios bins son no finitos).
    for (size_t i = 0; i < descriptores->size(); ++i) {
        // std::isfinite comprueba que no sea NaN ni Infinito
        if (std::isfinite(descriptores->points[i].histogram[0])) { //DEVUELVE TRUE SI NO ES NaN NI INFINITO
            desc_validos->push_back(descriptores->points[i]);
            kp_validos->push_back(keypoints_entrada->points[i]);
        }
    }
    // Sobrescribimos los keypoints originales para quitar los defectuosos
    *keypoints_entrada = *kp_validos; 

    return desc_validos;
}

// ---------------------------------------------------------
// MATCHING (Para FPFH)
// ---------------------------------------------------------

pcl::CorrespondencesPtr encontrarCorrespondencias(
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptores_source,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptores_target)
{
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> estimador_matching;
    estimador_matching.setInputSource(descriptores_source);
    estimador_matching.setInputTarget(descriptores_target);
    pcl::CorrespondencesPtr correspondencias(new pcl::Correspondences());
    estimador_matching.determineReciprocalCorrespondences(*correspondencias);
    return correspondencias;
}



// ---------------------------------------------------------
// CALCULAR TRANSFORMACIÓN (RANSAC)
// ---------------------------------------------------------
Eigen::Matrix4f calcularTransformacion(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_target,
    pcl::CorrespondencesPtr correspondencias)
{
    // Si el Matching fue un desastre y encontró muy pocas parejas, RANSAC puede crashear o 
    // inventarse un movimiento absurdo. Matemáticamente necesitamos al menos 3 para calcular un 
    // movimiento 3D, pero en la práctica pedimos más. Si no hay suficientes, devolvemos la 
    // "Matriz Identidad" (que significa: "El robot no se ha movido, quédate quieto").
    if (correspondencias->empty() || correspondencias->size() < MIN_CORRESPONDENCES_RANSAC) {
        std::cout << "WARN: Muy pocas correspondencias (" 
                  << correspondencias->size() << "), usando identidad" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> ransac;
    ransac.setInputSource(keypoints_source);
    ransac.setInputTarget(keypoints_target);
    ransac.setInputCorrespondences(correspondencias);
    // Inlier Threshold: El margen de error (en metros). Le decimos: "Si una pareja encaja a 
    // menos de 8cm de lo que dice la mayoría, acéptala como Inlier. Si se desvía más, es mentirosa".
    ransac.setInlierThreshold(RANSAC_INLIER_DIST);
    ransac.setMaximumIterations(RANSAC_MAX_ITER);

    pcl::Correspondences inliers;
    ransac.getRemainingCorrespondences(*correspondencias, inliers);

    std::cout << "RANSAC: " << inliers.size() << "/" << correspondencias->size() << " inliers" << std::endl;

    // Extraemos la transformación bruta calculada por RANSAC (El salto y giro en 3D libre).
    Eigen::Matrix4f transformacion_ransac = ransac.getBestTransformation();
    // A veces, incluso con RANSAC, el resultado es matemáticamente imposible (NaN) o se han 
    // quedado tan pocos inliers que no nos fiamos del resultado. Devolvemos Identidad.

    if (transformacion_ransac.hasNaN() || inliers.size() < MIN_INLIERS_TRANSFORM) {
        std::cout << "WARN: Transformacion invalida, usando identidad" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
    // Como la cámara del robot mira hacia adelante, su profundidad es Z y su lateral es X.
    // Calculamos exclusivamente cuánto ha rotado a izquierda o derecha (Yaw).

    float yaw = std::atan2(transformacion_ransac(0, 2), transformacion_ransac(0, 0));
    Eigen::Matrix4f constrained = Eigen::Matrix4f::Identity();
    constrained(0, 0) =  std::cos(yaw);
    constrained(0, 2) =  std::sin(yaw);
    constrained(2, 0) = -std::sin(yaw);
    constrained(2, 2) =  std::cos(yaw);
    // Ignoramos completamente cualquier salto en altura (Y)
    constrained(0, 3) = transformacion_ransac(0, 3);
    constrained(2, 3) = transformacion_ransac(2, 3);

    std::cout << "OK: yaw=" << yaw * 180.0f / M_PI
              << "°, tx=" << constrained(0, 3)
              << ", tz=" << constrained(2, 3) << std::endl;
    return constrained;
}




// ---------------------------------------------------------
// ALTERNATIVA 2: KEYPOINTS (HARRIS 3D)
// ---------------------------------------------------------
pcl::PointCloud<pcl::PointXYZRGB>::Ptr extraerKeypointsHarris(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_entrada) {
    // Harris usa el campo "Intensidad" para guardar temporalmente la "nota" o "fuerza" de la esquina.
    pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI> harris;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    tree->setInputCloud(cloud_entrada);
    harris.setSearchMethod(tree);    
    harris.setInputCloud(cloud_entrada);
    harris.setRadius(HARRIS_SALIENT_RADIUS);
    harris.setNonMaxSupression(true);
    harris.setThreshold(HARRIS_THRESHOLD); 
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_temp(new pcl::PointCloud<pcl::PointXYZI>);
    harris.compute(*keypoints_temp);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_final(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*keypoints_temp, *keypoints_final);
    return keypoints_final;
}
// ALTERNATIVA 2: DESCRIPTORES (PFH)
// ---------------------------------------------------------
// ---------------------------------------------------------
// DESCRIPTORES PFH
// ---------------------------------------------------------

pcl::PointCloud<pcl::PFHSignature125>::Ptr extraerDescriptoresPFH(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_entrada,
    pcl::PointCloud<pcl::Normal>::Ptr normales_finales,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_final) {
        
    pcl::PFHEstimation<pcl::PointXYZRGB ,pcl::Normal , pcl::PFHSignature125> pfh_estimador;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    pfh_estimador.setSearchMethod(tree);
    pfh_estimador.setInputCloud(keypoints_entrada);
    pfh_estimador.setSearchSurface(nube_final);
    pfh_estimador.setInputNormals(normales_finales);
    pfh_estimador.setRadiusSearch(PFH_RADIUS);
    
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptores(new pcl::PointCloud<pcl::PFHSignature125>);
    pfh_estimador.compute(*descriptores);

        //MISMO FILTRO ANTI-NAN Y INFINITOQUE EN FPFH
    pcl::PointCloud<pcl::PFHSignature125>::Ptr desc_validos(new pcl::PointCloud<pcl::PFHSignature125>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_validos(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < descriptores->size(); ++i) {
        if (std::isfinite(descriptores->points[i].histogram[0])) {
            desc_validos->push_back(descriptores->points[i]);
            kp_validos->push_back(keypoints_entrada->points[i]);
        }
    }
    // Sobrescribimos los keypoints originales para quitar los defectuosos
    *keypoints_entrada = *kp_validos;

    return desc_validos;
}
// ---------------------------------------------------------
// ALTERNATIVA 2: MATCHING (Para PFH)
// ---------------------------------------------------------
pcl::CorrespondencesPtr encontrarCorrespondenciasPFH(
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptores_source,
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptores_target) {
        
    pcl::registration::CorrespondenceEstimation<pcl::PFHSignature125, pcl::PFHSignature125> estimador_matching;
    estimador_matching.setInputSource(descriptores_source);
    estimador_matching.setInputTarget(descriptores_target);

    pcl::CorrespondencesPtr correspondencias(new pcl::Correspondences());
    estimador_matching.determineReciprocalCorrespondences(*correspondencias);

    return correspondencias;
}




class PclSubNode : public rclcpp::Node {
public:
    PclSubNode() : Node("sub_pcl") {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            TOPIC_NAME, QUEUE_SIZE,
            std::bind(&PclSubNode::callback, this, std::placeholders::_1));
    }

private:
    // -----------------------------------------------------------------------
    // FIX 3: El callback NO puede ser 'const' porque modifica variables
    // globales (keypoints_previos, descriptores_previos, mapa_global, etc.).
    // Declararlo 'const' es undefined behavior y puede causar crashes en
    // algunas plataformas/optimizaciones del compilador.
    // -----------------------------------------------------------------------
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // --- INICIA CRONÓMETRO TOTAL ---
        auto start_total = std::chrono::high_resolution_clock::now();

        // 1. Captura
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *cloud_raw);

        // 2. Limpieza NaN
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clean = filtrado(cloud_raw);

        // 3. Downsampling
        //VOXEL GRID
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> vGrid;
        vGrid.setInputCloud(cloud_clean);
        vGrid.setLeafSize(VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE);
        vGrid.filter(*cloud_downsampled);

        if (cloud_downsampled->empty()) return;

        // 4. Normales
        pcl::PointCloud<pcl::Normal>::Ptr normals = calcularNormales(cloud_downsampled);

        pcl::PointCloud<pcl::Normal>::Ptr normals_final(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<int> indices_validos;
        //FILTRADO DE PUNTOS CON NORMALES NO VALIDAS.(NAN)
        //indices_validos es un vector de índices de los puntos que tienen normales válidas
        pcl::removeNaNNormalsFromPointCloud(*normals, *normals_final, indices_validos);
        //copiamos los puntos de la nube downsampled que tienen normales válidas
        pcl::copyPointCloud(*cloud_downsampled, indices_validos, *cloud_final);
        //cloud_final es la nube final con los puntos que tienen normales válidas

        if (cloud_final->empty()) return;

        // ---------------------------------------------------------
        // 5. Keypoints 
        // exatremos los puntos que destacan en la nube, esquinas, bordes, etc.
        // ---------------------------------------------------------
        auto start_kp = std::chrono::high_resolution_clock::now();
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints;
        if (USAR_ISS) {
            keypoints = extraerKeypoints(cloud_final);
        } else {
            keypoints = extraerKeypointsHarris(cloud_final); 
            std::cout << "Keypoints HARRIS: " << keypoints->size() << std::endl;
        }
        
        auto end_kp = std::chrono::high_resolution_clock::now();
        double time_kp = std::chrono::duration<double, std::milli>(end_kp - start_kp).count();

        if (keypoints->empty()) return;

        // ---------------------------------------------------------
        // 6. Descriptores        
        // ---------------------------------------------------------
        auto start_desc = std::chrono::high_resolution_clock::now();
        
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptores_fpfh;
        pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptores_pfh;

        if (keypoints_previos->empty()) {
            std::cout << "Primer frame capturado. Inicializando memoria..." << std::endl;
            *mapa_global = *cloud_final;
            
            if (USAR_FPFH) descriptores_previos = extraerDescriptores(keypoints, normals_final, cloud_final);
            else descriptores_previos_pfh = extraerDescriptoresPFH(keypoints, normals_final, cloud_final);
            
            keypoints_previos = keypoints;


            // Guardamos la nube completa inicial para el ICP del próximo frame
            *nube_previa = *cloud_final; 
            return; // Salimos en el primer frame
        }

        // Extraemos descriptores del frame actual
        if (USAR_FPFH) descriptores_fpfh = extraerDescriptores(keypoints, normals_final, cloud_final);
        else descriptores_pfh = extraerDescriptoresPFH(keypoints, normals_final, cloud_final);
        
        auto end_desc = std::chrono::high_resolution_clock::now();
        double time_desc = std::chrono::duration<double, std::milli>(end_desc - start_desc).count();
// ---------------------------------------------------------
        // 7. Matching y RANSAC
        // ---------------------------------------------------------
        std::cout << "--- PROCESANDO MOVIMIENTO ---" << std::endl;
        
        auto start_match = std::chrono::high_resolution_clock::now();
        
        pcl::CorrespondencesPtr correspondencias;
        if (USAR_FPFH) {
            correspondencias = encontrarCorrespondencias(descriptores_previos, descriptores_fpfh);
            descriptores_previos = descriptores_fpfh; 
        } else {
            correspondencias = encontrarCorrespondenciasPFH(descriptores_previos_pfh, descriptores_pfh);
            std::cout << "Correspondencias PFH: " << correspondencias->size() << std::endl;
            descriptores_previos_pfh = descriptores_pfh; 
        }
        if (correspondencias->size() < MIN_CORRESPONDENCES_RANSAC) { //aqui habia 10
            std::cout << "Frame ignorado por pocas correspondencias: "
                      << correspondencias->size() << std::endl;
            
            // IMPORTANTE: actualizar memoria igualmente
            keypoints_previos = keypoints;

            //es un frame que se ha usado para calculos pero se descarta para el mapa global
            return;
        }

        
        //RANSAC
        Eigen::Matrix4f transformacion_frame = calcularTransformacion(keypoints_previos, keypoints, correspondencias);
        
        auto end_match = std::chrono::high_resolution_clock::now();
        double time_match = std::chrono::duration<double, std::milli>(end_match - start_match).count();

        // --- FILTRO DE SALTO Y GIRO IMPOSIBLE (Sanity Check Completo) ---
        float dx = transformacion_frame(0, 3); float dy = transformacion_frame(1, 3); float dz = transformacion_frame(2, 3);
        float distancia_salto = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        // Calculamos cuánto ha girado en el eje Yaw (en grados absolutos)
        float giro_grados = std::abs(std::atan2(transformacion_frame(0, 2), transformacion_frame(0, 0)) * 180.0f / M_PI);
        
        // Si salta más de 30cm o gira más de 45 grados de golpe, es un Falso Positivo por simetría
        if (distancia_salto > MAX_TRANSLATION_JUMP || giro_grados > MAX_ROTATION_JUMP) {
            std::cout << "WARN: FALSO POSITIVO IGNORADO: Salto de " << distancia_salto 
                      << "m y Giro de " << giro_grados << "°." << std::endl;
            transformacion_frame = Eigen::Matrix4f::Identity(); // Ignoramos el movimiento
        } else {
            std::cout << "Transformación detectada (Salto: " << distancia_salto 
                      << "m, Giro: " << giro_grados << "°):\n" << transformacion_frame << std::endl;
        }
        // -----------------------------------------------------------------


        // =========================================================
        // 8. REFINAMIENTO CON ICP (Iterative Closest Point)
        // =========================================================
        
        auto start_icp = std::chrono::high_resolution_clock::now();
        double time_icp = 0.0;

        // Solo ejecutamos ICP si RANSAC encontró un movimiento válido (no es la identidad pura)
        if (transformacion_frame != Eigen::Matrix4f::Identity()) {
            std::cout << "--- REFINANDO CON ICP ---" << std::endl;
            
            pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;

            // EL PORQUÉ DE ESTOS PARÁMETROS: Evitar cuellos de botella y falsos positivos
            icp.setMaximumIterations(35);             // Condición de parada 1: Máximo de iteraciones 
            icp.setTransformationEpsilon(1e-6);       // Condición de parada 2: El cambio en T es minúsculo 
            icp.setEuclideanFitnessEpsilon(1e-4);     // Condición de parada 3: El error cuadrático baja del umbral 
            icp.setMaxCorrespondenceDistance(0.05);   // Rechazar emparejamientos a más de 5cm (filtra ruido)

            // Configuramos las nubes: Actual (Source) contra Anterior (Target)
            icp.setInputSource(nube_previa);
            icp.setInputTarget(cloud_final);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_alineada_icp(new pcl::PointCloud<pcl::PointXYZRGB>());
            
            // LA CLAVE TÉCNICA: Pasamos la transformación del frame (RANSAC) como initial guess
            icp.align(*nube_alineada_icp, transformacion_frame);

            // Verificamos si ICP logró mejorar el resultado
            if (icp.hasConverged()) {
                std::cout << "ICP Convergió. Error residual (Fitness Score): " << icp.getFitnessScore() << std::endl;
                // Sobrescribimos la transformación bruta de RANSAC con la fina de ICP
                transformacion_frame = icp.getFinalTransformation();
            } else {
                std::cout << "WARN: ICP no convergió. Manteniendo transformación original de RANSAC." << std::endl;
            }

            auto end_icp = std::chrono::high_resolution_clock::now();
            time_icp = std::chrono::duration<double, std::milli>(end_icp - start_icp).count();
        }

        // --- ACTUALIZACIÓN DE MEMORIA FINAL ---
        keypoints_previos = keypoints;
        *nube_previa = *cloud_final; // Actualizamos la nube para el siguiente frame 
        // =========================================================



        transformacion_global = transformacion_global * transformacion_frame.inverse();



        pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_temp(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::transformPointCloud(*cloud_final, *nube_temp, transformacion_global);
        *mapa_global += *nube_temp;

        // Reducción del mapa
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapa_reducido(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> vg_map;
        vg_map.setInputCloud(mapa_global);
        vg_map.setLeafSize(VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE);
        vg_map.filter(*mapa_reducido);
        mapa_global = mapa_reducido;

        

        // Actualizamos visor
        {
            std::lock_guard<std::mutex> lock(visu_mutex);
            visu_pc = mapa_global;
        }

        // --- FINALIZA CRONÓMETRO TOTAL ---
        auto end_total = std::chrono::high_resolution_clock::now();
        double time_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();

        // ===================================================================
        // PANEL DE ESTADÍSTICAS 
        // ===================================================================
        std::cout << "\n================ ESTADÍSTICAS DEL FRAME ================" << std::endl;
        std::cout << "Puntos Procesados:   " << cloud_final->size() << std::endl;
        std::cout << "Keypoints Extraidos: " << keypoints->size() << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "T. Extracción KPs:   " << time_kp << " ms" << std::endl;
        std::cout << "T. Descriptores:     " << time_desc << " ms" << std::endl;
        std::cout << "T. Matching+RANSAC:  " << time_match << " ms" << std::endl;
        std::cout << "T. ICP:             " << time_icp << " ms" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "TIEMPO TOTAL FRAME:  " << time_total << " ms (~" << (1000.0/time_total) << " FPS)" << std::endl;
        std::cout << "========================================================\n" << std::endl;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
    // 1. MENÚ INTERACTIVO (Se ejecuta antes que ROS)
    std::cout << "\n========================================" << std::endl;
    std::cout << "   SISTEMA DE MAPEADO 3D (SLAM)         " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int opcion_kp, opcion_desc;
    
    std::cout << "\nElige el detector de Keypoints:" << std::endl;
    std::cout << "1. ISS3D (Rapido, por defecto)" << std::endl;
    std::cout << "2. Harris3D (Lento, esquinas fuertes)" << std::endl;
    std::cout << "Opcion (1 o 2): ";
    std::cin >> opcion_kp;
    USAR_ISS = (opcion_kp != 2); // Si elige 2 es false, si no, true

    std::cout << "\nElige el Descriptor:" << std::endl;
    std::cout << "1. FPFH (Rapido, por defecto)" << std::endl;
    std::cout << "2. PFH (Pesado, alta precision)" << std::endl;
    std::cout << "Opcion (1 o 2): ";
    std::cin >> opcion_desc;
    USAR_FPFH = (opcion_desc != 2);

    std::cout << "\n>>> INICIANDO NODO CON: " 
              << (USAR_ISS ? "ISS3D" : "Harris3D") << " + " 
              << (USAR_FPFH ? "FPFH" : "PFH") << " <<<\n" << std::endl;

    // 2. INICIO DE ROS2 Y PCL
    rclcpp::init(argc, argv);
    
    std::thread t(simpleVis); // Hilo del visualizador
    while (!viewer_started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    rclcpp::spin(std::make_shared<PclSubNode>()); // Bucle principal del robot
    
    rclcpp::shutdown();
    t.join();
    return 0;
}