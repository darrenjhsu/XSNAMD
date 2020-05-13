#include "expt_data.hh"

int num_q = 225;
int num_q2 = 256;
float q[225] = {0.015110062819394502, 0.017255509620332347, 0.019426920810090956, 0.021586397691660646, 0.023744995971951944, 0.02591500188562868, 0.028068119031097065, 0.03023795659603775, 0.03239546580484809, 0.034559083283400946, 0.03672249395960661, 0.03888197635553692, 0.04104724142166144, 0.043206872066762325, 0.04537070042534416, 0.04753226930982764, 0.04969427125321562, 0.051857203001850975, 0.0540184416543628, 0.05618157329625146, 0.05834304383320991, 0.06050566261807269, 0.06266775461812102, 0.06482981076235485, 0.06699229317845203, 0.0691541475656292, 0.07131671865704589, 0.07347849652349295, 0.07564116768584103, 0.07780290156701405, 0.07996539791545906, 0.08212766776151838, 0.08428925838791267, 0.08645262370039021, 0.08861326680333964, 0.09077704187214865, 0.09293817016901762, 0.09510037799541364, 0.0972640372901243, 0.09942327826024192, 0.10158951681576875, 0.10374731551948653, 0.10591325065650516, 0.10807326413101039, 0.11023344330745656, 0.11240238998631505, 0.114551357487752, 0.11673414913836701, 0.11886607678489051, 0.12108770173646827, 0.12346014465287904, 0.12565500974020227, 0.12779417683755423, 0.12997518694387736, 0.13212335803580966, 0.13429459847794795, 0.13645206786874725, 0.1386142962119586, 0.14077853497465592, 0.1429373141557634, 0.14510353268467954, 0.14726234391931664, 0.14942661512724362, 0.15158810494819308, 0.15374974472164804, 0.15591319620452299, 0.15807410362289126, 0.16023744459223588, 0.16239885570291665, 0.16456128292525218, 0.16672360372157016, 0.1688855565066247, 0.17104819966801382, 0.17320993132094148, 0.17537239750665304, 0.1775344091273128, 0.17969661656409294, 0.18185904688964918, 0.1840210625940345, 0.1861833529331529, 0.1883454384182867, 0.19050761430433438, 0.1926698662517989, 0.1948321378325587, 0.1969943396378297, 0.19915644850421493, 0.2013186664225372, 0.20348077461837466, 0.2056430541800956, 0.20780535194567143, 0.20996747376610597, 0.21212967016648807, 0.214291799381493, 0.21645402424955312, 0.21861634777032227, 0.22077850645488697, 0.2229406629698635, 0.22510284350831936, 0.2272649737384265, 0.22942722614495598, 0.23158952977524205, 0.23375168339075253, 0.23591384129639167, 0.23807602000989891, 0.24023815274295204, 0.2424004233263771, 0.24456271583494063, 0.2467248547965875, 0.24888702275461783, 0.25104919603129994, 0.25321132929243156, 0.25537362753602155, 0.2575358937354739, 0.2596980285619234, 0.2618602116924041, 0.2640223483513595, 0.26618459010286966, 0.26834690474219874, 0.270509050022259, 0.27267121248540854, 0.27483339016526553, 0.2769955205678675, 0.27915779563006043, 0.2813200837681634, 0.28348222515698035, 0.28564439376181333, 0.28780656250837394, 0.2899687039994675, 0.29213099206439946, 0.29429326734446576, 0.2964553974690492, 0.29861757502246505, 0.3007797381393286, 0.3029418826970648, 0.30510419767539937, 0.30726643957727146, 0.30942857579673405, 0.3115907600470735, 0.31375289372816534, 0.31591516043831763, 0.31807745833716466, 0.3202395914789708, 0.32240176569437257, 0.3245639315616628, 0.3267260732928317, 0.3288883627394847, 0.3310506357431575, 0.3332127685691685, 0.3353749442315438, 0.33753710814597343, 0.3396992525444705, 0.3418615669525935, 0.34402380973030594, 0.3461859450687333, 0.3483481293117377, 0.3505102636454247, 0.35267252980768743, 0.3548348283014271, 0.3569969604324355, 0.3591591351173588, 0.3613213024073976, 0.3634834413964236, 0.36564573469416045, 0.36780800430413374, 0.369970138413742, 0.3721323152054423, 0.37429447487671325, 0.3764566276760059, 0.37861893089092347, 0.3807811856453191, 0.3829433113284228, 0.38510549759542106, 0.38726764866452956, 0.38942981039127916, 0.391592135480593, 0.39375435343849996, 0.39591649484346475, 0.39807867480404896, 0.40024081253811583, 0.40240310043962835, 0.40456537876732357, 0.4067275041380451, 0.40888968722184676, 0.4110518440527919, 0.41321399618550403, 0.41537630275012727, 0.4175385522357409, 0.4197006828798496, 0.42186286651683347, 0.4240250170301047, 0.42618718453653204, 0.4283494987141702, 0.4305117309340839, 0.432673857062466, 0.43483604910682333, 0.4369981889287672, 0.4391603719795076, 0.44132270104930404, 0.44348489611586656, 0.44564704499125957, 0.4478092196358587, 0.44997136546986366, 0.4521336700472508, 0.45429592637229654, 0.4564580486349564, 0.4586202390581324, 0.46078238548303546, 0.46294455268308227, 0.46510687208099116, 0.46726909654254495, 0.4694312304038987, 0.4715934169256541, 0.47375555720750473, 0.47591774800658004, 0.4780800598598571, 0.48024227952604337, 0.48240440084673913, 0.4845665992184203, 0.4867287349371598, 0.4888909264406416, 0.4910532772660535, 0.49321542370978716, 0.4953776066562572, 0.4975397601961581, 0.4997019090508193};
float S_exp[225] = {55.02432772074472, 53.897757534076455, 52.43423485587447, 51.20035905260355, 49.79719622564566, 48.317703161367305, 46.60896536100269, 44.88126165217619, 43.10853029562138, 41.396293784605106, 39.606033714776835, 37.83245538794987, 36.15258613645464, 34.20270956582934, 32.39532417206692, 30.64141771978549, 28.94497324237573, 27.577499262312575, 25.9732511415698, 24.27623279005975, 22.816959556749637, 21.315626800254645, 19.795966736731835, 18.391469387976002, 17.03373078314288, 15.7284866700638, 14.48013212885612, 13.241420495177218, 12.145892874034654, 11.155341459408467, 10.138626402574502, 9.191845721773758, 8.36134545190783, 7.585918331866471, 6.861663450708078, 6.222752662152951, 5.646215486336204, 5.1203106106963405, 4.638949884328393, 4.203977392166, 3.837930063788482, 3.4838261546777782, 3.166593542500598, 2.897526294454506, 2.6495123326780115, 2.454764667433056, 2.2656364667590885, 2.0981305304658298, 1.9508847142575967, 1.8194645293108476, 1.7310219474795807, 1.620720589629589, 1.5205072816135876, 1.4412736455070319, 1.3698756233002793, 1.3200646819761903, 1.2616152159728609, 1.2129852872117983, 1.1650588181380823, 1.1178880692036401, 1.0805878589051379, 1.0389207492586066, 1.0106426317365775, 0.9782107559314285, 0.9361474930777491, 0.9077932253724127, 0.8799937818889455, 0.8468202133459635, 0.8101653866467404, 0.7791769243859227, 0.7590284727410935, 0.735618706123757, 0.7080068070521501, 0.6758413785674356, 0.6507023824085223, 0.6283986710012027, 0.6060816859175844, 0.5905601162692015, 0.5603758614440084, 0.5364435863825534, 0.5265445035537428, 0.5075037813436208, 0.48495845080700645, 0.4615204967007172, 0.4504606717552191, 0.4439192243226181, 0.4198572320223582, 0.40586737709712795, 0.39638680807198334, 0.38048357193680254, 0.37013240782346435, 0.3553085344163632, 0.3502977284809381, 0.3332901595914375, 0.3137781644273049, 0.3079261298687378, 0.2936882680208862, 0.29128113410887485, 0.2796876365581166, 0.2647754639108178, 0.25470382645452394, 0.24970132106060725, 0.26163222901507904, 0.24559940065220465, 0.23224377505135896, 0.22929791505759217, 0.21281186727471454, 0.20988319140603345, 0.21152729658390507, 0.2116819342423431, 0.2020601703459657, 0.19452346069432622, 0.19606585095381357, 0.19195415262357013, 0.18935387626432137, 0.18048094071162404, 0.18613810759990373, 0.19495407243724142, 0.18142739688845932, 0.18943352834459365, 0.19446713330263413, 0.18260449617294452, 0.1857680122776053, 0.18564735577354796, 0.1911005307916473, 0.1883469174947015, 0.1799235003258583, 0.19418077019980648, 0.186027697521196, 0.18229029752310813, 0.19948276354274022, 0.19280334768090757, 0.1924526839223275, 0.19681012112988214, 0.19105300111797818, 0.18636932407821094, 0.18675557286862976, 0.18726394351910064, 0.18656168136094406, 0.18992713675438203, 0.18097127792983067, 0.17763775639370782, 0.18162248312150367, 0.1815427764629211, 0.17956070331491636, 0.1673362996990712, 0.16877529635086674, 0.16737828734408364, 0.15914007907316433, 0.16451293977831766, 0.16549701006511672, 0.16489757716352846, 0.16137380408147436, 0.16273827992768863, 0.16523714020705796, 0.1517026616950761, 0.1523363015243044, 0.1596073973864231, 0.15179310961079234, 0.14746343476501783, 0.1489860762421068, 0.14836842841958067, 0.14540817243862353, 0.14418947979185523, 0.14665141562574172, 0.14970107528991766, 0.14551766973141392, 0.14051807429829638, 0.1453935865805046, 0.15128457509040635, 0.150549722401437, 0.1484813047595701, 0.14947954527519525, 0.14860154463391265, 0.15025843927718652, 0.15142630783615646, 0.1477064011535503, 0.15117493635328694, 0.15913493099906548, 0.1517869316865214, 0.14763130922621887, 0.15561732524922645, 0.1544381215269153, 0.1591043393736207, 0.15935915857858002, 0.15822453117386912, 0.1643396151632461, 0.1589750591909777, 0.1593187096218474, 0.15454745366916456, 0.15090350303005642, 0.16434811417196157, 0.16054848639754496, 0.1639862963868954, 0.16599928529387514, 0.14835489615579342, 0.15519416810213432, 0.16268841881735982, 0.15619337259784744, 0.15603786697812713, 0.1605129024038909, 0.16517441389072487, 0.15734116378252816, 0.15720738696624817, 0.1678600842933138, 0.1587368008196926, 0.1587742980304502, 0.16413558385555616, 0.16485010573773065, 0.17286788568563188, 0.15978456626865653, 0.1535055545944713, 0.16346315982757742, 0.1617212803541253, 0.15758872456688627, 0.15853246571328422, 0.1639698696436006, 0.16505530052065462, 0.16311412570903908, 0.16505423525361199, 0.16285553062847055, 0.16556203145410223, 0.17595281257668707, 0.1658116637952576, 0.16012723160861994};
float S_err[225] = {0.3245092631388218, 0.24011286325239795, 0.325663184280214, 0.15799634045943736, 0.29693460953316436, 0.18033693716956523, 0.23403230003469308, 0.1976763912700803, 0.17595090902116498, 0.20516085104127374, 0.14986753733987837, 0.1848948596257748, 0.149472697351752, 0.15282344386618113, 0.14853060878589888, 0.12873732376577138, 0.13781200774409286, 0.11779941292711259, 0.12126809047851908, 0.10993057148062818, 0.10361555037675536, 0.10141678321553382, 0.09003738139764089, 0.08933860296681291, 0.08031768281795909, 0.07680480700436156, 0.07174784275620091, 0.06547982534876605, 0.06286677761968236, 0.05773193997635924, 0.05547724042983862, 0.051388632552150704, 0.048426568520282376, 0.04590931244819319, 0.0425817616307054, 0.04080955527634822, 0.03819068529617222, 0.036405293314935645, 0.03458600022627869, 0.03271981466113336, 0.031593600923364856, 0.0299739676805261, 0.02884193756921315, 0.027723498189647342, 0.026655394774252387, 0.026049656552205664, 0.024996009890920344, 0.024705029676247766, 0.02361822469955694, 0.02397459907583075, 0.02745656280587369, 0.026227373751302664, 0.023023894562457774, 0.02240322048651971, 0.021789294425293388, 0.021603821245937224, 0.021268166763718, 0.02095289757871117, 0.020828912611335532, 0.02047812897278407, 0.020079440569768087, 0.01967610650962324, 0.01961930816970596, 0.01944810234913279, 0.019298176094004724, 0.019177322924089534, 0.018967645506669878, 0.01886199575278108, 0.01866521315289765, 0.018578987629638305, 0.01847587732089131, 0.018324820783669964, 0.018225920473042573, 0.01804153689593142, 0.0179316491163837, 0.01785958215321474, 0.017769544621649692, 0.017657563965949363, 0.017525244424403886, 0.017382627238637757, 0.017276439101764563, 0.017209403038310075, 0.0170657736600393, 0.016994163511549216, 0.016936979852829697, 0.016882014545160053, 0.016817681580096423, 0.016737014714723816, 0.01670136933867403, 0.01657293597908973, 0.01652099070774768, 0.016432617279315494, 0.01635529970747058, 0.016297671460265773, 0.016175143971663725, 0.01617969236859714, 0.016038648900512514, 0.01604521505297697, 0.015954248940092185, 0.015791508280112835, 0.01584326048288538, 0.01575436820070866, 0.015855959232682147, 0.015644830994194142, 0.015675616096711847, 0.01555424742844221, 0.015451380775290444, 0.015592693067270114, 0.015206912181690927, 0.01582975309769281, 0.014753887785258678, 0.01746334694049979, 0.022609693197768423, 0.01893337806156056, 0.016159885098065378, 0.016888152737671736, 0.016078839493735262, 0.01599586704876949, 0.015485843995958402, 0.015536841948423022, 0.015598569018737487, 0.015403910534018834, 0.015556049217436977, 0.015383395435364764, 0.015502362395196515, 0.015394086207132245, 0.015282050343002685, 0.015388598850871498, 0.015313599888512401, 0.015345075124483111, 0.015304328367793598, 0.015223645937265532, 0.015253024319183918, 0.015238692129280086, 0.01518777752641661, 0.015142550963979524, 0.015156929716606829, 0.015103524760502415, 0.01504884697565258, 0.015099921501536077, 0.015051537916338295, 0.01504171231407067, 0.015031506876771558, 0.014955518263428688, 0.014981388348255234, 0.014938768639771892, 0.014908955291210982, 0.014898123060883567, 0.014828872216409425, 0.014848687177348949, 0.014819367431464368, 0.014809189982469618, 0.014786516448076801, 0.01473403961339, 0.014783798519039304, 0.01468241234689015, 0.01470714214856422, 0.014709736546873962, 0.014589545720739715, 0.014693485921441075, 0.014602294364109572, 0.014673494592439386, 0.014613965793810961, 0.014587980096911805, 0.014689234155156591, 0.014504082242397364, 0.014778187614108226, 0.014395836281812697, 0.014677558383622214, 0.014567635524835653, 0.01443013250139536, 0.015009481385419295, 0.0136134935288632, 0.018629701976358663, 0.021589949326394804, 0.016124451671814866, 0.015533211744868533, 0.015201502124592934, 0.014247487296541836, 0.014853041722283244, 0.0143356894307758, 0.014620375802723294, 0.014481480350281073, 0.014489144398983186, 0.014570375161668111, 0.014429464189176936, 0.014539306118222146, 0.014368658248964111, 0.014443944746781514, 0.014342043400384268, 0.01430524137870507, 0.014371462153544854, 0.01425451050069721, 0.014315215913872312, 0.014259527554276284, 0.014233381829540702, 0.014237756912417, 0.014144780558398646, 0.01416866479729829, 0.014177217202545778, 0.014180529719871151, 0.014183285183307626, 0.014145282461986037, 0.014176251875766923, 0.014180578078284604, 0.01410549421401777, 0.014020635010663548, 0.014000462517686593, 0.014130294227941298, 0.014171406242864934, 0.014085332598523853, 0.014091276246374539, 0.014079540907947214, 0.014116753088210499, 0.014078962019415064, 0.014046068833358403, 0.01413491085414736, 0.01405532637803445, 0.014091267392692867, 0.014056065715699714, 0.014055756696607262, 0.014113833621072292, 0.014076166395703976, 0.014172035550685044, 0.014002156843047802};
