#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

namespace spectrum {

constexpr float kUndefinedWavelength = -1.0f;

constexpr uint32_t RGBResponseShortestWavelength = 390u;
constexpr uint32_t RGBResponseLongestWavelength = 780u;
constexpr uint32_t RGBResponseWavelengthCount = RGBResponseLongestWavelength - RGBResponseShortestWavelength + 1u;

constexpr uint32_t ShortestWavelength = 390u;
constexpr uint32_t WavelengthCount = 441u;
constexpr uint32_t LongestWavelength = ShortestWavelength + WavelengthCount - 1u;

constexpr float kShortestWavelength = static_cast<float>(ShortestWavelength);
constexpr float kLongestWavelength = static_cast<float>(LongestWavelength);
constexpr float kWavelengthCount = static_cast<float>(WavelengthCount);

constexpr float kRGBResponseShortestWavelength = static_cast<float>(RGBResponseShortestWavelength);
constexpr float kRGBResponseLongestWavelength = static_cast<float>(RGBResponseLongestWavelength);
constexpr float kRGBResponseWavelengthCount = static_cast<float>(RGBResponseWavelengthCount);

constexpr const float3 spectral_xyz(uint32_t i) {
  ETX_ASSERT(i < WavelengthCount);

  constexpr const float3 kCIE2006[WavelengthCount] = {{0.00295242f, 0.0004076779f, 0.01318752f}, {0.003577275f, 0.0004977769f, 0.01597879f},
    {0.004332146f, 0.0006064754f, 0.01935758f}, {0.005241609f, 0.000737004f, 0.02343758f}, {0.006333902f, 0.0008929388f, 0.02835021f}, {0.007641137f, 0.001078166f, 0.03424588f},
    {0.009199401f, 0.001296816f, 0.04129467f}, {0.01104869f, 0.001553159f, 0.04968641f}, {0.01323262f, 0.001851463f, 0.05962964f}, {0.01579791f, 0.002195795f, 0.07134926f},
    {0.01879338f, 0.002589775f, 0.08508254f}, {0.02226949f, 0.003036799f, 0.1010753f}, {0.02627978f, 0.003541926f, 0.1195838f}, {0.03087862f, 0.004111422f, 0.1408647f},
    {0.0361189f, 0.004752618f, 0.1651644f}, {0.04204986f, 0.005474207f, 0.1927065f}, {0.04871256f, 0.006285034f, 0.2236782f}, {0.05612868f, 0.007188068f, 0.2582109f},
    {0.06429866f, 0.008181786f, 0.2963632f}, {0.07319818f, 0.009260417f, 0.3381018f}, {0.08277331f, 0.01041303f, 0.3832822f}, {0.09295327f, 0.01162642f, 0.4316884f},
    {0.1037137f, 0.01289884f, 0.483244f}, {0.115052f, 0.01423442f, 0.5379345f}, {0.1269771f, 0.0156408f, 0.595774f}, {0.1395127f, 0.01712968f, 0.6568187f},
    {0.1526661f, 0.01871265f, 0.7210459f}, {0.1663054f, 0.02038394f, 0.7878635f}, {0.1802197f, 0.02212935f, 0.8563391f}, {0.1941448f, 0.02392985f, 0.9253017f},
    {0.2077647f, 0.02576133f, 0.9933444f}, {0.2207911f, 0.02760156f, 1.059178f}, {0.2332355f, 0.02945513f, 1.122832f}, {0.2452462f, 0.03133884f, 1.184947f},
    {0.2570397f, 0.03327575f, 1.246476f}, {0.2688989f, 0.03529554f, 1.308674f}, {0.2810677f, 0.03742705f, 1.372628f}, {0.2933967f, 0.03967137f, 1.437661f},
    {0.3055933f, 0.04201998f, 1.502449f}, {0.3173165f, 0.04446166f, 1.565456f}, {0.3281798f, 0.04698226f, 1.62494f}, {0.3378678f, 0.04956742f, 1.679488f},
    {0.3465097f, 0.05221219f, 1.729668f}, {0.3543953f, 0.05491387f, 1.776755f}, {0.3618655f, 0.05766919f, 1.822228f}, {0.3693084f, 0.06047429f, 1.867751f},
    {0.3770107f, 0.06332195f, 1.914504f}, {0.384685f, 0.06619271f, 1.961055f}, {0.3918591f, 0.06906185f, 2.005136f}, {0.3980192f, 0.0719019f, 2.044296f},
    {0.4026189f, 0.07468288f, 2.075946f}, {0.4052637f, 0.07738452f, 2.098231f}, {0.4062482f, 0.08003601f, 2.112591f}, {0.406066f, 0.08268524f, 2.121427f},
    {0.4052283f, 0.08538745f, 2.127239f}, {0.4042529f, 0.08820537f, 2.132574f}, {0.4034808f, 0.09118925f, 2.139093f}, {0.4025362f, 0.09431041f, 2.144815f},
    {0.4008675f, 0.09751346f, 2.146832f}, {0.3979327f, 0.1007349f, 2.14225f}, {0.3932139f, 0.103903f, 2.128264f}, {0.3864108f, 0.1069639f, 2.103205f},
    {0.3779513f, 0.1099676f, 2.069388f}, {0.3684176f, 0.1129992f, 2.03003f}, {0.3583473f, 0.1161541f, 1.988178f}, {0.3482214f, 0.1195389f, 1.946651f},
    {0.338383f, 0.1232503f, 1.907521f}, {0.3288309f, 0.1273047f, 1.870689f}, {0.3194977f, 0.1316964f, 1.835578f}, {0.3103345f, 0.1364178f, 1.801657f},
    {0.3013112f, 0.1414586f, 1.76844f}, {0.2923754f, 0.1468003f, 1.735338f}, {0.2833273f, 0.1524002f, 1.701254f}, {0.2739463f, 0.1582021f, 1.665053f},
    {0.2640352f, 0.16414f, 1.625712f}, {0.2534221f, 0.1701373f, 1.582342f}, {0.2420135f, 0.1761233f, 1.534439f}, {0.2299346f, 0.1820896f, 1.482544f},
    {0.2173617f, 0.1880463f, 1.427438f}, {0.2044672f, 0.1940065f, 1.369876f}, {0.1914176f, 0.1999859f, 1.310576f}, {0.1783672f, 0.2060054f, 1.250226f},
    {0.1654407f, 0.2120981f, 1.189511f}, {0.1527391f, 0.2183041f, 1.12905f}, {0.1403439f, 0.2246686f, 1.069379f}, {0.1283167f, 0.2312426f, 1.010952f},
    {0.1167124f, 0.2380741f, 0.9541809f}, {0.1056121f, 0.2451798f, 0.8995253f}, {0.09508569f, 0.2525682f, 0.847372f}, {0.08518206f, 0.2602479f, 0.7980093f},
    {0.0759312f, 0.2682271f, 0.7516389f}, {0.06733159f, 0.2765005f, 0.7082645f}, {0.05932018f, 0.2850035f, 0.6673867f}, {0.05184106f, 0.2936475f, 0.6284798f},
    {0.04486119f, 0.3023319f, 0.5911174f}, {0.0383677f, 0.3109438f, 0.5549619f}, {0.03237296f, 0.3194105f, 0.5198843f}, {0.02692095f, 0.3278683f, 0.4862772f},
    {0.0220407f, 0.3365263f, 0.4545497f}, {0.01773951f, 0.3456176f, 0.4249955f}, {0.01400745f, 0.3554018f, 0.3978114f}, {0.01082291f, 0.3660893f, 0.3730218f},
    {0.008168996f, 0.3775857f, 0.3502618f}, {0.006044623f, 0.389696f, 0.3291407f}, {0.004462638f, 0.4021947f, 0.3093356f}, {0.00344681f, 0.4148227f, 0.2905816f},
    {0.003009513f, 0.4273539f, 0.2726773f}, {0.003090744f, 0.4398206f, 0.2555143f}, {0.003611221f, 0.452336f, 0.2390188f}, {0.004491435f, 0.4650298f, 0.2231335f},
    {0.005652072f, 0.4780482f, 0.2078158f}, {0.007035322f, 0.4915173f, 0.1930407f}, {0.008669631f, 0.5054224f, 0.1788089f}, {0.01060755f, 0.5197057f, 0.1651287f},
    {0.01290468f, 0.5343012f, 0.1520103f}, {0.01561956f, 0.5491344f, 0.1394643f}, {0.0188164f, 0.5641302f, 0.1275353f}, {0.02256923f, 0.5792416f, 0.1163771f},
    {0.02694456f, 0.5944264f, 0.1061161f}, {0.0319991f, 0.6096388f, 0.09682266f}, {0.03778185f, 0.6248296f, 0.08852389f}, {0.04430635f, 0.6399656f, 0.08118263f},
    {0.05146516f, 0.6550943f, 0.07463132f}, {0.05912224f, 0.6702903f, 0.06870644f}, {0.0671422f, 0.6856375f, 0.06327834f}, {0.07538941f, 0.7012292f, 0.05824484f},
    {0.08376697f, 0.7171103f, 0.05353812f}, {0.09233581f, 0.7330917f, 0.04914863f}, {0.101194f, 0.7489041f, 0.04507511f}, {0.1104362f, 0.764253f, 0.04131175f},
    {0.1201511f, 0.7788199f, 0.03784916f}, {0.130396f, 0.792341f, 0.03467234f}, {0.141131f, 0.804851f, 0.03175471f}, {0.1522944f, 0.8164747f, 0.02907029f},
    {0.1638288f, 0.827352f, 0.02659651f}, {0.1756832f, 0.8376358f, 0.02431375f}, {0.1878114f, 0.8474653f, 0.02220677f}, {0.2001621f, 0.8568868f, 0.02026852f},
    {0.2126822f, 0.8659242f, 0.01849246f}, {0.2253199f, 0.8746041f, 0.01687084f}, {0.2380254f, 0.8829552f, 0.01539505f}, {0.2507787f, 0.8910274f, 0.0140545f},
    {0.2636778f, 0.8989495f, 0.01283354f}, {0.2768607f, 0.9068753f, 0.01171754f}, {0.2904792f, 0.9149652f, 0.01069415f}, {0.3046991f, 0.9233858f, 0.009753f},
    {0.3196485f, 0.9322325f, 0.008886096f}, {0.3352447f, 0.9412862f, 0.008089323f}, {0.351329f, 0.9502378f, 0.007359131f}, {0.3677148f, 0.9587647f, 0.006691736f},
    {0.3841856f, 0.9665325f, 0.006083223f}, {0.4005312f, 0.9732504f, 0.005529423f}, {0.4166669f, 0.9788415f, 0.005025504f}, {0.432542f, 0.9832867f, 0.004566879f},
    {0.4481063f, 0.986572f, 0.004149405f}, {0.4633109f, 0.9886887f, 0.003769336f}, {0.478144f, 0.9897056f, 0.003423302f}, {0.4927483f, 0.9899849f, 0.003108313f},
    {0.5073315f, 0.9899624f, 0.00282165f}, {0.5221315f, 0.9900731f, 0.00256083f}, {0.537417f, 0.99075f, 0.002323578f}, {0.5534217f, 0.9922826f, 0.002107847f},
    {0.5701242f, 0.9943837f, 0.001911867f}, {0.5874093f, 0.9966221f, 0.001734006f}, {0.6051269f, 0.9985649f, 0.001572736f}, {0.6230892f, 0.9997775f, 0.001426627f},
    {0.6410999f, 0.999944f, 0.001294325f}, {0.6590659f, 0.99922f, 0.001174475f}, {0.6769436f, 0.9978793f, 0.001065842f}, {0.6947143f, 0.9961934f, 0.0009673215f},
    {0.7123849f, 0.9944304f, 0.0008779264f}, {0.7299978f, 0.9927831f, 0.0007967847f}, {0.7476478f, 0.9911578f, 0.0007231502f}, {0.765425f, 0.9893925f, 0.0006563501f},
    {0.7834009f, 0.9873288f, 0.0005957678f}, {0.8016277f, 0.9848127f, 0.0005408385f}, {0.8201041f, 0.9817253f, 0.0004910441f}, {0.8386843f, 0.9780714f, 0.0004459046f},
    {0.8571936f, 0.973886f, 0.0004049826f}, {0.8754652f, 0.9692028f, 0.0003678818f}, {0.8933408f, 0.9640545f, 0.0003342429f}, {0.9106772f, 0.9584409f, 0.0003037407f},
    {0.9273554f, 0.9522379f, 0.0002760809f}, {0.9432502f, 0.9452968f, 0.000250997f}, {0.9582244f, 0.9374773f, 0.0002282474f}, {0.9721304f, 0.9286495f, 0.0002076129f},
    {0.9849237f, 0.9187953f, 0.0001888948f}, {0.9970067f, 0.9083014f, 0.0001719127f}, {1.008907f, 0.8976352f, 0.000156503f}, {1.021163f, 0.8872401f, 0.0001425177f},
    {1.034327f, 0.877536f, 0.000129823f}, {1.048753f, 0.868792f, 0.0001182974f}, {1.063937f, 0.8607474f, 0.000107831f}, {1.079166f, 0.8530233f, 9.832455E-05f},
    {1.093723f, 0.8452535f, 8.968787E-05f}, {1.106886f, 0.8370838f, 8.183954E-05f}, {1.118106f, 0.8282409f, 7.470582E-05f}, {1.127493f, 0.818732f, 6.821991E-05f},
    {1.135317f, 0.8086352f, 6.232132E-05f}, {1.141838f, 0.7980296f, 5.695534E-05f}, {1.147304f, 0.786995f, 5.207245E-05f}, {1.151897f, 0.775604f, 4.762781E-05f},
    {1.155582f, 0.7638996f, 4.358082E-05f}, {1.158284f, 0.7519157f, 3.989468E-05f}, {1.159934f, 0.7396832f, 3.653612E-05f}, {1.160477f, 0.7272309f, 3.347499E-05f},
    {1.15989f, 0.7145878f, 3.0684E-05f}, {1.158259f, 0.7017926f, 2.813839E-05f}, {1.155692f, 0.6888866f, 2.581574E-05f}, {1.152293f, 0.6759103f, 2.369574E-05f},
    {1.148163f, 0.6629035f, 2.175998E-05f}, {1.143345f, 0.6498911f, 1.999179E-05f}, {1.137685f, 0.636841f, 1.837603E-05f}, {1.130993f, 0.6237092f, 1.689896E-05f},
    {1.123097f, 0.6104541f, 1.554815E-05f}, {1.113846f, 0.5970375f, 1.431231E-05f}, {1.103152f, 0.5834395f, 1.318119E-05f}, {1.091121f, 0.5697044f, 1.214548E-05f},
    {1.077902f, 0.5558892f, 1.119673E-05f}, {1.063644f, 0.5420475f, 1.032727E-05f}, {1.048485f, 0.5282296f, 9.53013E-06f}, {1.032546f, 0.5144746f, 8.798979E-06f},
    {1.01587f, 0.5007881f, 8.128065E-06f}, {0.9984859f, 0.4871687f, 7.51216E-06f}, {0.9804227f, 0.473616f, 6.946506E-06f}, {0.9617111f, 0.4601308f, 6.426776E-06f},
    {0.9424119f, 0.446726f, 0.0f}, {0.9227049f, 0.4334589f, 0.0f}, {0.9027804f, 0.4203919f, 0.0f}, {0.8828123f, 0.407581f, 0.0f}, {0.8629581f, 0.3950755f, 0.0f},
    {0.8432731f, 0.3828894f, 0.0f}, {0.8234742f, 0.370919f, 0.0f}, {0.8032342f, 0.3590447f, 0.0f}, {0.7822715f, 0.3471615f, 0.0f}, {0.7603498f, 0.3351794f, 0.0f},
    {0.7373739f, 0.3230562f, 0.0f}, {0.713647f, 0.3108859f, 0.0f}, {0.6895336f, 0.298784f, 0.0f}, {0.6653567f, 0.2868527f, 0.0f}, {0.6413984f, 0.2751807f, 0.0f},
    {0.6178723f, 0.2638343f, 0.0f}, {0.5948484f, 0.252833f, 0.0f}, {0.57236f, 0.2421835f, 0.0f}, {0.5504353f, 0.2318904f, 0.0f}, {0.5290979f, 0.2219564f, 0.0f},
    {0.5083728f, 0.2123826f, 0.0f}, {0.4883006f, 0.2031698f, 0.0f}, {0.4689171f, 0.1943179f, 0.0f}, {0.4502486f, 0.185825f, 0.0f}, {0.4323126f, 0.1776882f, 0.0f},
    {0.415079f, 0.1698926f, 0.0f}, {0.3983657f, 0.1623822f, 0.0f}, {0.3819846f, 0.1550986f, 0.0f}, {0.3657821f, 0.1479918f, 0.0f}, {0.3496358f, 0.1410203f, 0.0f},
    {0.3334937f, 0.1341614f, 0.0f}, {0.3174776f, 0.1274401f, 0.0f}, {0.3017298f, 0.1208887f, 0.0f}, {0.2863684f, 0.1145345f, 0.0f}, {0.27149f, 0.1083996f, 0.0f},
    {0.2571632f, 0.1025007f, 0.0f}, {0.2434102f, 0.09684588f, 0.0f}, {0.2302389f, 0.09143944f, 0.0f}, {0.2176527f, 0.08628318f, 0.0f}, {0.2056507f, 0.08137687f, 0.0f},
    {0.1942251f, 0.07671708f, 0.0f}, {0.183353f, 0.07229404f, 0.0f}, {0.1730097f, 0.06809696f, 0.0f}, {0.1631716f, 0.06411549f, 0.0f}, {0.1538163f, 0.06033976f, 0.0f},
    {0.144923f, 0.05676054f, 0.0f}, {0.1364729f, 0.05336992f, 0.0f}, {0.1284483f, 0.05016027f, 0.0f}, {0.120832f, 0.04712405f, 0.0f}, {0.1136072f, 0.04425383f, 0.0f},
    {0.1067579f, 0.04154205f, 0.0f}, {0.1002685f, 0.03898042f, 0.0f}, {0.09412394f, 0.03656091f, 0.0f}, {0.08830929f, 0.03427597f, 0.0f}, {0.0828101f, 0.03211852f, 0.0f},
    {0.07761208f, 0.03008192f, 0.0f}, {0.07270064f, 0.02816001f, 0.0f}, {0.06806167f, 0.02634698f, 0.0f}, {0.06368176f, 0.02463731f, 0.0f}, {0.05954815f, 0.02302574f, 0.0f},
    {0.05564917f, 0.02150743f, 0.0f}, {0.05197543f, 0.02007838f, 0.0f}, {0.04851788f, 0.01873474f, 0.0f}, {0.04526737f, 0.01747269f, 0.0f}, {0.04221473f, 0.01628841f, 0.0f},
    {0.03934954f, 0.01517767f, 0.0f}, {0.0366573f, 0.01413473f, 0.0f}, {0.03412407f, 0.01315408f, 0.0f}, {0.03173768f, 0.01223092f, 0.0f}, {0.02948752f, 0.01136106f, 0.0f},
    {0.02736717f, 0.0105419f, 0.0f}, {0.02538113f, 0.00977505f, 0.0f}, {0.02353356f, 0.009061962f, 0.0f}, {0.02182558f, 0.008402962f, 0.0f}, {0.0202559f, 0.007797457f, 0.0f},
    {0.01881892f, 0.00724323f, 0.0f}, {0.0174993f, 0.006734381f, 0.0f}, {0.01628167f, 0.006265001f, 0.0f}, {0.01515301f, 0.005830085f, 0.0f}, {0.0141023f, 0.005425391f, 0.0f},
    {0.01312106f, 0.005047634f, 0.0f}, {0.01220509f, 0.00469514f, 0.0f}, {0.01135114f, 0.004366592f, 0.0f}, {0.01055593f, 0.004060685f, 0.0f}, {0.009816228f, 0.00377614f, 0.0f},
    {0.009128517f, 0.003511578f, 0.0f}, {0.008488116f, 0.003265211f, 0.0f}, {0.007890589f, 0.003035344f, 0.0f}, {0.007332061f, 0.002820496f, 0.0f},
    {0.006809147f, 0.002619372f, 0.0f}, {0.006319204f, 0.00243096f, 0.0f}, {0.005861036f, 0.002254796f, 0.0f}, {0.005433624f, 0.002090489f, 0.0f},
    {0.005035802f, 0.001937586f, 0.0f}, {0.004666298f, 0.001795595f, 0.0f}, {0.00432375f, 0.001663989f, 0.0f}, {0.004006709f, 0.001542195f, 0.0f},
    {0.003713708f, 0.001429639f, 0.0f}, {0.003443294f, 0.001325752f, 0.0f}, {0.003194041f, 0.00122998f, 0.0f}, {0.002964424f, 0.001141734f, 0.0f},
    {0.002752492f, 0.001060269f, 0.0f}, {0.002556406f, 0.0009848854f, 0.0f}, {0.002374564f, 0.0009149703f, 0.0f}, {0.002205568f, 0.0008499903f, 0.0f},
    {0.002048294f, 0.0007895158f, 0.0f}, {0.001902113f, 0.0007333038f, 0.0f}, {0.001766485f, 0.0006811458f, 0.0f}, {0.001640857f, 0.0006328287f, 0.0f},
    {0.001524672f, 0.0005881375f, 0.0f}, {0.001417322f, 0.0005468389f, 0.0f}, {0.001318031f, 0.0005086349f, 0.0f}, {0.001226059f, 0.0004732403f, 0.0f},
    {0.001140743f, 0.0004404016f, 0.0f}, {0.001061495f, 0.0004098928f, 0.0f}, {0.0009877949f, 0.0003815137f, 0.0f}, {0.0009191847f, 0.0003550902f, 0.0f},
    {0.0008552568f, 0.0003304668f, 0.0f}, {0.0007956433f, 0.000307503f, 0.0f}, {0.000740012f, 0.0002860718f, 0.0f}, {0.000688098f, 0.0002660718f, 0.0f},
    {0.0006397864f, 0.0002474586f, 0.0f}, {0.0005949726f, 0.0002301919f, 0.0f}, {0.0005535291f, 0.0002142225f, 0.0f}, {0.0005153113f, 0.0001994949f, 0.0f},
    {0.0004801234f, 0.0001859336f, 0.0f}, {0.0004476245f, 0.0001734067f, 0.0f}, {0.0004174846f, 0.0001617865f, 0.0f}, {0.0003894221f, 0.0001509641f, 0.0f},
    {0.0003631969f, 0.0001408466f, 0.0f}, {0.0003386279f, 0.0001313642f, 0.0f}, {0.0003156452f, 0.0001224905f, 0.0f}, {0.0002941966f, 0.000114206f, 0.0f},
    {0.0002742235f, 0.0001064886f, 0.0f}, {0.0002556624f, 9.931439E-05f, 0.0f}, {0.000238439f, 9.265512E-05f, 0.0f}, {0.0002224525f, 8.647225E-05f, 0.0f},
    {0.0002076036f, 8.07278E-05f, 0.0f}, {0.0001938018f, 7.538716E-05f, 0.0f}, {0.0001809649f, 7.041878E-05f, 0.0f}, {0.0001690167f, 6.579338E-05f, 0.0f},
    {0.0001578839f, 6.14825E-05f, 0.0f}, {0.0001474993f, 5.746008E-05f, 0.0f}, {0.0001378026f, 5.370272E-05f, 0.0f}, {0.0001287394f, 5.018934E-05f, 0.0f},
    {0.0001202644f, 4.690245E-05f, 0.0f}, {0.0001123502f, 4.383167E-05f, 0.0f}, {0.0001049725f, 4.09678E-05f, 0.0f}, {9.810596E-05f, 3.830123E-05f, 0.0f},
    {9.172477E-05f, 3.582218E-05f, 0.0f}, {8.579861E-05f, 3.351903E-05f, 0.0f}, {8.028174E-05f, 3.137419E-05f, 0.0f}, {7.513013E-05f, 2.937068E-05f, 0.0f},
    {7.030565E-05f, 2.74938E-05f, 0.0f}, {6.577532E-05f, 2.573083E-05f, 0.0f}, {6.151508E-05f, 2.407249E-05f, 0.0f}, {5.752025E-05f, 2.251704E-05f, 0.0f},
    {5.378813E-05f, 2.10635E-05f, 0.0f}, {5.03135E-05f, 1.970991E-05f, 0.0f}, {4.708916E-05f, 1.845353E-05f, 0.0f}, {4.410322E-05f, 1.728979E-05f, 0.0f},
    {4.13315E-05f, 1.620928E-05f, 0.0f}, {3.874992E-05f, 1.520262E-05f, 0.0f}, {3.633762E-05f, 1.426169E-05f, 0.0f}, {3.407653E-05f, 1.337946E-05f, 0.0f},
    {3.195242E-05f, 1.255038E-05f, 0.0f}, {2.995808E-05f, 1.177169E-05f, 0.0f}, {2.808781E-05f, 1.104118E-05f, 0.0f}, {2.633581E-05f, 1.035662E-05f, 0.0f},
    {2.46963E-05f, 9.715798E-06f, 0.0f}, {2.316311E-05f, 9.116316E-06f, 0.0f}, {2.172855E-05f, 8.555201E-06f, 0.0f}, {2.038519E-05f, 8.029561E-06f, 0.0f},
    {1.912625E-05f, 7.536768E-06f, 0.0f}, {1.794555E-05f, 7.074424E-06f, 0.0f}, {1.683776E-05f, 6.640464E-06f, 0.0f}, {1.579907E-05f, 6.233437E-06f, 0.0f},
    {1.482604E-05f, 5.852035E-06f, 0.0f}, {1.391527E-05f, 5.494963E-06f, 0.0f}, {1.306345E-05f, 5.160948E-06f, 0.0f}, {1.22672E-05f, 4.848687E-06f, 0.0f},
    {1.152279E-05f, 4.556705E-06f, 0.0f}, {1.082663E-05f, 4.28358E-06f, 0.0f}, {1.01754E-05f, 4.027993E-06f, 0.0f}, {9.565993E-06f, 3.788729E-06f, 0.0f},
    {8.995405E-06f, 3.564599E-06f, 0.0f}, {8.460253E-06f, 3.354285E-06f, 0.0f}, {7.957382E-06f, 3.156557E-06f, 0.0f}, {7.483997E-06f, 2.970326E-06f, 0.0f},
    {7.037621E-06f, 2.794625E-06f, 0.0f}, {6.616311E-06f, 2.628701E-06f, 0.0f}, {6.219265E-06f, 2.472248E-06f, 0.0f}, {5.845844E-06f, 2.32503E-06f, 0.0f},
    {5.495311E-06f, 2.186768E-06f, 0.0f}, {5.166853E-06f, 2.057152E-06f, 0.0f}, {4.859511E-06f, 1.935813E-06f, 0.0f}, {4.571973E-06f, 1.822239E-06f, 0.0f},
    {4.30292E-06f, 1.715914E-06f, 0.0f}, {4.051121E-06f, 1.616355E-06f, 0.0f}, {3.815429E-06f, 1.523114E-06f, 0.0f}, {3.594719E-06f, 1.43575E-06f, 0.0f},
    {3.387736E-06f, 1.353771E-06f, 0.0f}, {3.193301E-06f, 1.276714E-06f, 0.0f}, {3.010363E-06f, 1.204166E-06f, 0.0f}, {2.83798E-06f, 1.135758E-06f, 0.0f},
    {2.675365E-06f, 1.071181E-06f, 0.0f}, {2.52202E-06f, 1.010243E-06f, 0.0f}, {2.377511E-06f, 9.527779E-07f, 0.0f}, {2.241417E-06f, 8.986224E-07f, 0.0f},
    {2.113325E-06f, 8.476168E-07f, 0.0f}, {1.99283E-06f, 7.996052E-07f, 0.0f}, {1.879542E-06f, 7.544361E-07f, 0.0f}, {1.773083E-06f, 7.119624E-07f, 0.0f},
    {1.673086E-06f, 6.720421E-07f, 0.0f}, {1.579199E-06f, 6.34538E-07f, 0.0f}};

  return kCIE2006[i];
}

ETX_GPU_CODE float3 xyz_to_rgb(const float3& xyz) {
  return {
    3.24045420f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
    -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
    0.05564340f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z,
  };
}

ETX_GPU_CODE float3 rgb_to_xyz(const float3& rgb) {
  return {
    0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f + rgb.z,
    0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f + rgb.z,
    0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f + rgb.z,
  };
}

ETX_GPU_CODE float4 rgb_to_xyz4(const float3& rgb) {
  return {
    0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z,
    0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z,
    0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z,
    1.0f,
  };
}

ETX_GPU_CODE float black_body_radiation_maximum_wavelength(float t_kelvins) {
  return 2.8977729e+6f / t_kelvins;
}

ETX_GPU_CODE float black_body_radiation(float wavelength_nm, float t_kelvins) {
  ETX_ASSERT(t_kelvins > 0);

  // wavelength (in nm) is scaled to reduce floating point errors, constants are scaled correspondingly
  constexpr float wavelengt_scale = 1.0f / 1000.0f;
  constexpr float Lc1 = 3.7417712e+5f;  // 2 * pi * h * c * c * (10^21 - from wavelength scale)
  constexpr float Lc2 = 1.4387752e+4f;  // h * c / k * (10^-6 - from wavelength scale)

  wavelength_nm *= wavelengt_scale;
  float wl5 = wavelength_nm * (wavelength_nm * wavelength_nm) * (wavelength_nm * wavelength_nm);

  float e0 = expf(Lc2 / (wavelength_nm * t_kelvins));
  ETX_ASSERT(isnan(e0) == false);

  float d = wl5 * (e0 - 1.0f);
  ETX_ASSERT(isnan(d) == false);

  return isinf(d) ? 0.0f : (Lc1 / d);
}

constexpr const float kYIntegral() {
  float result = 0.0f;
  for (uint32_t i = 0; i < WavelengthCount; ++i) {
    result += spectral_xyz(i).y;
  }
  return result;
}

}  // namespace spectrum

struct SpectralQuery {
  enum : uint32_t {
    Spectral = 1u << 0u,
  };
  float wavelength = spectrum::kUndefinedWavelength;
  uint32_t flags = 0u;

  SpectralQuery() = default;

  SpectralQuery(float w, const uint32_t& f)
    : wavelength(w)
    , flags(f) {
  }

  bool spectral() const {
    return (flags & Spectral) != 0;
  };

  float sampling_pdf() const {
    return spectral() ? 1.0f / spectrum::kWavelengthCount : 1.0f;
  }

  bool valid() const {
    return (wavelength >= spectrum::kShortestWavelength) && (wavelength <= spectrum::kLongestWavelength);
  }

  static SpectralQuery sample() {
    return SpectralQuery{
      spectrum::kUndefinedWavelength,
      0u,
    };
  }

  static SpectralQuery spectral_sample(float rnd) {
    return SpectralQuery{
      spectrum::kShortestWavelength + rnd * (spectrum::kLongestWavelength - spectrum::kShortestWavelength),
      Spectral,
    };
  }
};

struct ETX_ALIGNED SpectralValue {
  float3 integrated = {};
  float w = 0.0f;
};

struct ETX_ALIGNED SpectralResponse : public SpectralQuery {
  SpectralValue components = {};

  SpectralResponse() = default;

  SpectralResponse(const SpectralQuery q)
    : SpectralQuery(q) {
  }

  SpectralResponse(const SpectralQuery q, float value)
    : SpectralQuery(q)
    , components{{value, value, value}, value} {
  }

  SpectralResponse(const SpectralQuery q, const float3& c)
    : SpectralQuery(q)
    , components{c} {
  }

  float component_count() const {
    return spectral() ? 1.0f : 3.0f;
  }

  const SpectralQuery& query() const {
    return *this;
  }

  ETX_GPU_CODE float3 to_xyz() const {
    if (spectral() == false) {
      return spectrum::rgb_to_xyz(components.integrated);
    }

    if ((components.w == 0.0f) || (wavelength < spectrum::kShortestWavelength) || (wavelength > spectrum::kLongestWavelength))
      return {};

    constexpr float kYScale = 1.0f / spectrum::kYIntegral();

    ETX_ASSERT(valid());
    float w = floorf(wavelength);
    float dw = wavelength - w;
    uint32_t i = static_cast<uint32_t>(w - spectrum::kShortestWavelength);
    uint32_t j = min(i + 1u, spectrum::WavelengthCount - 1u);
    float3 xyz0 = spectrum::spectral_xyz(i);
    float3 xyz1 = spectrum::spectral_xyz(j);
    return lerp<float3>(xyz0, xyz1, dw) * (components.w * kYScale);
  }

  ETX_GPU_CODE float3 to_rgb() const {
    return spectral() ? spectrum::xyz_to_rgb(to_xyz()) : components.integrated;
  }

  ETX_GPU_CODE float minimum() const {
    return spectral() ? components.w : min(components.integrated.x, min(components.integrated.y, components.integrated.z));
  }

  ETX_GPU_CODE float maximum() const {
    return spectral() ? components.w : max(components.integrated.x, max(components.integrated.y, components.integrated.z));
  }

  ETX_GPU_CODE float monochromatic() const {
    return spectral() ? components.w : components.integrated.y;
  }

  ETX_GPU_CODE float sum() const {
    return spectral() ? components.w : components.integrated.x + components.integrated.y + components.integrated.z;
  }

  ETX_GPU_CODE float average() const {
    return spectral() ? components.w : (components.integrated.x + components.integrated.y + components.integrated.z) / 3.0f;
  }

  ETX_GPU_CODE float component(uint32_t i) const {
    ETX_ASSERT(i < 3);
    return spectral() ? components.w : *(&components.integrated.x + i);
  }

  ETX_GPU_CODE bool valid() const {
    return spectral() ? valid_value(components.w) : valid_value(components.integrated);
  }

  ETX_GPU_CODE bool is_zero() const {
    return spectral() ? (components.w <= kEpsilon) : (components.integrated.x <= kEpsilon) && (components.integrated.y <= kEpsilon) && (components.integrated.z <= kEpsilon);
  }

#define SPECTRAL_OP(OP)                                                        \
  ETX_GPU_CODE SpectralResponse& operator OP(const SpectralResponse & other) { \
    ETX_ASSERT_EQUAL(wavelength, other.wavelength);                            \
    components.integrated OP other.components.integrated;                      \
    components.w OP other.components.w;                                        \
    return *this;                                                              \
  }
  SPECTRAL_OP(+=)
  SPECTRAL_OP(-=)
  SPECTRAL_OP(*=)
  SPECTRAL_OP(/=)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                                                                                                                                  \
  ETX_GPU_CODE SpectralResponse operator OP(const SpectralResponse& other) const {                                                                                       \
    ETX_ASSERT_EQUAL(wavelength, other.wavelength);                                                                                                                      \
    ETX_ASSERT((spectral() && other.spectral()) || ((spectral() == false) && (other.spectral() == false)));                                                              \
    return spectral() ? SpectralResponse{query(), components.w OP other.components.w} : SpectralResponse{query(), components.integrated OP other.components.integrated}; \
  }
  SPECTRAL_OP(+)
  SPECTRAL_OP(-)
  SPECTRAL_OP(*)
  SPECTRAL_OP(/)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                     \
  ETX_GPU_CODE SpectralResponse& operator OP(float other) { \
    components.integrated OP other;                         \
    components.w OP other;                                  \
    return *this;                                           \
  }
  SPECTRAL_OP(+=)
  SPECTRAL_OP(-=)
  SPECTRAL_OP(*=)
  SPECTRAL_OP(/=)
#undef SPECTRAL_OP

#define SPECTRAL_OP(OP)                                                                                                               \
  ETX_GPU_CODE SpectralResponse operator OP(float other) const {                                                                      \
    return spectral() ? SpectralResponse{query(), components.w OP other} : SpectralResponse{query(), components.integrated OP other}; \
  }
  SPECTRAL_OP(+)
  SPECTRAL_OP(-)
  SPECTRAL_OP(*)
  SPECTRAL_OP(/)
#undef SPECTRAL_OP
};

ETX_GPU_CODE bool valid_value(const SpectralResponse& v) {
  return v.valid();
}
ETX_GPU_CODE SpectralResponse operator*(float other, const SpectralResponse& s) {
  return s * other;
}
ETX_GPU_CODE SpectralResponse operator/(float other, const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), other / s.components.w} : SpectralResponse{s.query(), other / s.components.integrated};
}
ETX_GPU_CODE SpectralResponse operator+(float other, const SpectralResponse& s) {
  return s + other;
}
ETX_GPU_CODE SpectralResponse operator-(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), -s.components.w} : SpectralResponse{s.query(), -s.components.integrated};
}
ETX_GPU_CODE SpectralResponse operator-(float other, const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), other - s.components.w} : SpectralResponse{s.query(), other - s.components.integrated};
}
ETX_GPU_CODE SpectralResponse exp(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), expf(s.components.w)} : SpectralResponse{s.query(), exp(s.components.integrated)};
}
ETX_GPU_CODE SpectralResponse sqrt(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), sqrtf(s.components.w)} : SpectralResponse{s.query(), sqrt(s.components.integrated)};
}
ETX_GPU_CODE SpectralResponse cos(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), cosf(s.components.w)} : SpectralResponse{s.query(), cos(s.components.integrated)};
}
ETX_GPU_CODE SpectralResponse abs(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), fabsf(s.components.w)} : SpectralResponse{s.query(), abs(s.components.integrated)};
}
ETX_GPU_CODE SpectralResponse saturate(const SpectralResponse& s) {
  return s.spectral() ? SpectralResponse{s.query(), saturate(s.components.w)} : SpectralResponse{s.query(), saturate(s.components.integrated)};
}

#if (ETX_DEBUG || ETX_FORCE_VALIDATION)
template <class T>
ETX_GPU_CODE void print_invalid_value(const char* name, const T& v, const char* filename, uint32_t line);

template <>
ETX_GPU_CODE void print_invalid_value<float>(const char* name, const float& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f) at %s [%u]\n", name, v, filename, line);
}

template <>
ETX_GPU_CODE void print_invalid_value<float2>(const char* name, const float2& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f) at %s [%u]\n", name, v.x, v.y, filename, line);
}

template <>
ETX_GPU_CODE void print_invalid_value<float3>(const char* name, const float3& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, filename, line);
}

template <>
ETX_GPU_CODE void print_invalid_value<float4>(const char* name, const float4& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f %f %f %f) at %s [%u]\n", name, v.x, v.y, v.z, v.w, filename, line);
}

template <>
ETX_GPU_CODE void print_invalid_value<complex>(const char* name, const complex& z, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f + i * %f) at %s [%u]\n", name, z.real(), z.imag(), filename, line);
}

template <>
ETX_GPU_CODE void print_invalid_value<SpectralResponse>(const char* name, const SpectralResponse& v, const char* filename, uint32_t line) {
  printf("Validation failed: %s (%f : %f %f %f / %f) at %s [%u]\n", name, v.wavelength, v.components.integrated.x, v.components.integrated.y, v.components.integrated.z,
    v.components.w, filename, line);
}
#endif

struct Spectrums;

struct ETX_ALIGNED SpectralDistribution {
  enum Class : uint32_t {
    Invalid,
    Reflectance,
    Conductor,
    Dielectric,
    Illuminant,
  };

  enum Mapping : uint32_t {
    Direct,
    Color,
  };

  struct {
    float wavelength = 0.0f;
    float power = 0.0f;
  } spectral_entries[spectrum::WavelengthCount] = {};

  uint32_t spectral_entry_count = 0u;

 public:  // device
  ETX_GPU_CODE SpectralResponse query(const SpectralQuery q) const {
    if (q.spectral() == false) {
      return SpectralResponse{q, integrated_value};
    }

    ETX_ASSERT(q.valid());

    auto lower_bound = [this](float wavelength) {
      uint32_t b = 0;
      uint32_t e = spectral_entry_count;
      do {
        uint32_t m = b + (e - b) / 2;
        if (spectral_entries[m].wavelength > wavelength) {
          e = m;
        } else {
          b = m;
        }
      } while ((e - b) > 1);
      return b;
    };

    uint32_t i = lower_bound(q.wavelength);
    if (i >= spectral_entry_count) {
      return {q, 0.0f};
    }

    if ((i == 0) && (q.wavelength < spectral_entries[i].wavelength)) {
      return {q, 0.0f};
    }

    if ((i + 1 == spectral_entry_count) && (q.wavelength > spectral_entries[i].wavelength)) {
      return {q, 0.0f};
    }

    uint32_t j = min(i + 1u, spectral_entry_count - 1);
    float t = (i == j) ? 0.0f : (q.wavelength - spectral_entries[i].wavelength) / (spectral_entries[j].wavelength - spectral_entries[i].wavelength);
    float p = lerp(spectral_entries[i].power, spectral_entries[j].power, t);
    ETX_VALIDATE(p);
    return SpectralResponse{q, p};
  }

  ETX_GPU_CODE SpectralResponse operator()(const SpectralQuery q) const {
    return query(q);
  }

  ETX_GPU_CODE bool empty() const {
    return spectral_entry_count == 0;
  }

  ETX_GPU_CODE bool is_zero() const {
    for (uint32_t i = 0; i < spectral_entry_count; ++i) {
      if (spectral_entries[i].power != 0.0f) {
        return false;
      }
    }
    return true;
  }

 public:
  SpectralDistribution() = default;

  void scale(float factor);

  float3 integrate_to_xyz() const;

  const float3& integrated() const;

  float luminance() const;
  float maximum_spectral_power() const;

  bool valid() const;

  static SpectralDistribution from_samples(const float2 wavelengths_power[], uint64_t count, Mapping mapping);

  static SpectralDistribution null();
  static SpectralDistribution constant(float value);
  static SpectralDistribution from_black_body(float temperature, float scale);
  static SpectralDistribution from_normalized_black_body(float temperature, float scale);
  static SpectralDistribution rgb(const float3& rgb);

  static Class load_from_file(const char* file_name, SpectralDistribution& values0, SpectralDistribution* values1, bool extend_range, Mapping mapping);

 private:
  float3 integrated_value = {};
};

struct RefractiveIndex {
  SpectralDistribution eta;
  SpectralDistribution k;

  struct Sample : public SpectralQuery {
    SpectralResponse eta;
    SpectralResponse k;

    ETX_GPU_CODE complex as_complex_x() const {
      ETX_ASSERT(spectral() == false);
      return complex{eta.components.integrated.x, k.components.integrated.x};
    }

    ETX_GPU_CODE complex as_complex_y() const {
      ETX_ASSERT(spectral() == false);
      return {eta.components.integrated.y, k.components.integrated.y};
    }

    ETX_GPU_CODE complex as_complex_z() const {
      ETX_ASSERT(spectral() == false);
      return {eta.components.integrated.z, k.components.integrated.z};
    }

    ETX_GPU_CODE complex as_complex() const {
      ETX_ASSERT(spectral());
      return {eta.components.w, k.components.w};
    }

    ETX_GPU_CODE complex as_monochromatic_complex() const {
      return {eta.monochromatic(), k.monochromatic()};
    }
  };

  ETX_GPU_CODE Sample at(SpectralQuery q) const {
    Sample result = {q};
    result.eta = eta.empty() ? SpectralResponse(q, 1.0f) : eta(q);
    result.k = k.empty() ? SpectralResponse(q, 0.0f) : k(q);
    return result;
  }

  ETX_GPU_CODE Sample operator()(const SpectralQuery q) const {
    return at(q);
  }
};

struct ETX_ALIGNED Spectrums {
  RefractiveIndex thinfilm = {};
  RefractiveIndex conductor = {};
  RefractiveIndex dielectric = {};
  SpectralDistribution rayleigh = {};
  SpectralDistribution mie = {};
  SpectralDistribution ozone = {};
  SpectralDistribution black = {};
};

namespace spectrum {

Pointer<Spectrums> shared();

}

namespace rgb {

ETX_GPU_CODE SpectralDistribution make_spd(const float3& rgb) {
  return SpectralDistribution::rgb(rgb);
}

ETX_GPU_CODE SpectralResponse query_spd(const SpectralQuery spect, const float3& rgb) {
  ETX_ASSERT(spect.spectral());

  constexpr float3 response[spectrum::RGBResponseWavelengthCount] = {{0.361396471056708f, 0.252275705864828f, 0.386327749991032f},
    {0.366205305492837f, 0.235416148479571f, 0.398378488359465f}, {0.371266544491276f, 0.21551362909132f, 0.413219780889496f},
    {0.375826682752906f, 0.193361838306436f, 0.430811443451517f}, {0.378931281646137f, 0.170213950722939f, 0.450854739893322f},
    {0.379486367262762f, 0.146935008502323f, 0.473578602657459f}, {0.376593730760743f, 0.124520895668156f, 0.498885356527868f},
    {0.369558349313344f, 0.104513543161356f, 0.525928094462577f}, {0.358873584245906f, 0.0873768593817381f, 0.553749546116267f},
    {0.345130566986974f, 0.072963535226373f, 0.581905889627497f}, {0.328251563088163f, 0.0610398371212503f, 0.610708593091079f},
    {0.308552505415631f, 0.0513190921255156f, 0.640128397106951f}, {0.287174624808842f, 0.0433949906128264f, 0.669430380270105f},
    {0.265201641787191f, 0.0368826097572261f, 0.6979157450652f}, {0.243045727135127f, 0.0314748187061981f, 0.725479451241526f},
    {0.220937878555078f, 0.0269368109732676f, 0.752125308174961f}, {0.199346669367302f, 0.023102088919068f, 0.777551239751982f},
    {0.178921019256518f, 0.0198804141079897f, 0.801198565198056f}, {0.160082581682908f, 0.0171951627504054f, 0.822722254635495f},
    {0.143043624509629f, 0.0149725918056479f, 0.8419837825555f}, {0.127647149539569f, 0.0131454383663785f, 0.859207411299804f},
    {0.114788573984292f, 0.0116487280786844f, 0.873562697296838f}, {0.103316260235184f, 0.0104086932344483f, 0.886275046348654f},
    {0.0932518666479813f, 0.00936480867269029f, 0.897383323885414f}, {0.0843664327916331f, 0.00847145272002132f, 0.907162114032132f},
    {0.0764577020390705f, 0.00769379991480418f, 0.915848497678197f}, {0.0693845001234432f, 0.00700864335902012f, 0.923606856599652f},
    {0.0631104736138383f, 0.00641054210726977f, 0.930478984200588f}, {0.0576046106519274f, 0.00589653172568395f, 0.936498857804f},
    {0.0528265500146353f, 0.00546274898698706f, 0.941710700798877f}, {0.0487277455283648f, 0.00510516507487266f, 0.946167089538723f},
    {0.0452464572772848f, 0.00481770982409582f, 0.949935832819348f}, {0.0422725820425418f, 0.00458608307901361f, 0.953141334919228f},
    {0.039703262373669f, 0.00439668646497515f, 0.955900051146983f}, {0.037455139058216f, 0.0042384011939596f, 0.95830645982211f},
    {0.0354600372812668f, 0.00410184676768847f, 0.960438115944202f}, {0.0336564467190598f, 0.00397915803696281f, 0.962364395327574f},
    {0.0319714709061577f, 0.00386435524834813f, 0.964164173954034f}, {0.0303366881445415f, 0.00375206176074797f, 0.965911250170638f},
    {0.0286933099515374f, 0.00363728832521505f, 0.967669401788626f}, {0.0269947205432021f, 0.00351577644591095f, 0.969489502991052f},
    {0.0252282217111559f, 0.00338580522402117f, 0.971385973151492f}, {0.0234580083647742f, 0.00325171858372909f, 0.973290273119263f},
    {0.0217527228115435f, 0.00311853091617314f, 0.975128746288891f}, {0.0201631925905153f, 0.00299005839993127f, 0.976846749143154f},
    {0.0187234576407893f, 0.00286884779294434f, 0.978407694612511f}, {0.0174488994506994f, 0.00275663952405543f, 0.979794461148403f},
    {0.0163280904902336f, 0.00265565645819332f, 0.981016253259442f}, {0.0153452468886244f, 0.00256813127759055f, 0.982086621981104f},
    {0.0144861780559633f, 0.00249614943952171f, 0.983017672674219f}, {0.0137381956551172f, 0.00244178408164589f, 0.983820020431351f},
    {0.0130867474600425f, 0.00240592043900152f, 0.984507332309904f}, {0.0125066008328427f, 0.00238445312895018f, 0.985108946336329f},
    {0.011974039235965f, 0.00237241835583257f, 0.985653542707892f}, {0.0114695448056021f, 0.00236512543691745f, 0.986165330037658f},
    {0.0109771084042056f, 0.00235800437504581f, 0.986664887523997f}, {0.0104892403205561f, 0.00234831284715694f, 0.987162447120292f},
    {0.0100203951870777f, 0.00234037259593541f, 0.9876392324978f}, {0.00958680322480087f, 0.00234017145989162f, 0.988073025622264f},
    {0.00920042585973419f, 0.00235367021686863f, 0.98844590421961f}, {0.00887973072081534f, 0.00238786516006445f, 0.988732404442236f},
    {0.00860520827065324f, 0.00244643706709479f, 0.988948354934752f}, {0.00836732063173359f, 0.00252780336726841f, 0.989104876266163f},
    {0.00817777671420739f, 0.00263166056927542f, 0.98919056298573f}, {0.00797728196919423f, 0.00275031422799895f, 0.989272404046558f},
    {0.00776280756773801f, 0.00288050327073746f, 0.98935668943516f}, {0.00752322480656702f, 0.00301827558713578f, 0.989458499794256f},
    {0.00738176284055312f, 0.00320753131923758f, 0.989410706072419f}, {0.00717964596439533f, 0.00336752685268753f, 0.989452827341526f},
    {0.00696721407329869f, 0.00356965698692528f, 0.989463129090929f}, {0.00716956338982263f, 0.00393592724088271f, 0.988894509503006f},
    {0.00648723991847956f, 0.00404960604105882f, 0.989463154193344f}, {0.00621620145572442f, 0.00432111336519908f, 0.989462685306371f},
    {0.0061703417349609f, 0.00479375415814202f, 0.989035904178019f}, {0.00563348899865069f, 0.00495835307356082f, 0.989408158002329f},
    {0.00523483096796861f, 0.00531690945791252f, 0.989448259605118f}, {0.00498377479547626f, 0.0058246166160886f, 0.989191608599878f},
    {0.004760911210281f, 0.00647607419329812f, 0.98876301457901f}, {0.00454208151498636f, 0.00727757698303369f, 0.988180341484035f},
    {0.00432748461853859f, 0.00827234652401926f, 0.987400168773758f}, {0.00411781951832206f, 0.00952326464963346f, 0.986358915732748f},
    {0.00391410266312994f, 0.01112695996682f, 0.984958937279945f}, {0.00371757718278664f, 0.0132432651492109f, 0.983039157613706f},
    {0.00352930218980754f, 0.0161528616582408f, 0.980317836116817f}, {0.00334999643313325f, 0.0203955708927266f, 0.976254432608402f},
    {0.00317999229993085f, 0.0271578607388973f, 0.969662146952061f}, {0.00301966868590795f, 0.0396332002480259f, 0.957347131162118f},
    {0.0028705041211961f, 0.070325853543585f, 0.926803642017144f}, {0.00273363515083904f, 0.24992097853778f, 0.747345386062212f},
    {0.00260950322849179f, 0.879787924982616f, 0.117602571707971f}, {0.00249856306030856f, 0.942808059071281f, 0.0546933780587161f},
    {0.00240066290421108f, 0.961783902277145f, 0.0358154347561766f}, {0.00231441511884631f, 0.970867923905408f, 0.0268176608120465f},
    {0.002238381861103f, 0.976185165981655f, 0.0215764518969246f}, {0.00217134831811772f, 0.979669430483185f, 0.0181592209856099f},
    {0.00211227274835322f, 0.982122997849185f, 0.0157647290494959f}, {0.0020599209693597f, 0.983940230816847f, 0.0139998478832392f},
    {0.00201189235285397f, 0.985344314099166f, 0.0126437931961052f}, {0.00196564633691464f, 0.986472843092402f, 0.0115615102294631f},
    {0.0019188261125259f, 0.987415571260503f, 0.0106656022184081f}, {0.00186926818990606f, 0.988233933976807f, 0.00989679742413786f},
    {0.00181574188025131f, 0.988967351292169f, 0.00921690639569143f}, {0.00175994912822348f, 0.989628436356831f, 0.00861161406580099f},
    {0.00170413548536508f, 0.99022167636531f, 0.00807418769585527f}, {0.00165032161433526f, 0.990749529062014f, 0.00760014882948853f},
    {0.00160029995498627f, 0.991213264076789f, 0.00718643549381231f}, {0.00155523718724574f, 0.9916158354081f, 0.00682892692078097f},
    {0.00151452891335767f, 0.991968465695081f, 0.00651700491485177f}, {0.00147724647237568f, 0.992282013932627f, 0.0062407391000766f},
    {0.00144256240580218f, 0.992565087190795f, 0.00599234991916047f}, {0.00140973348628126f, 0.992824627795095f, 0.00576563823354133f},
    {0.00137819444959272f, 0.993065715524064f, 0.00555608953555029f}, {0.00134785214360529f, 0.993290265822737f, 0.0053618815464683f},
    {0.00131872380811099f, 0.993499408376993f, 0.00518186733421665f}, {0.00129082419404661f, 0.993694174648865f, 0.0050150006772275f},
    {0.00126416553960866f, 0.993875508465271f, 0.00486032552149924f}, {0.00123879678964318f, 0.994043994807604f, 0.00471720794432022f},
    {0.00121490507172813f, 0.994199227128935f, 0.00458586734744344f}, {0.00119268748968416f, 0.99434082146648f, 0.00446649061921878f},
    {0.00117231872567021f, 0.9944685990092f, 0.00435908185244806f}, {0.00115395914029295f, 0.99458250229289f, 0.00426353773101575f},
    {0.00113765908123584f, 0.994682942820404f, 0.00417939769621543f}, {0.00112308682297844f, 0.994772367251123f, 0.00410454554247749f},
    {0.00110983780104095f, 0.994853284357552f, 0.00403687747816938f}, {0.00109753225709246f, 0.994927910792738f, 0.00397455660214574f},
    {0.0010858089015939f, 0.994998227701555f, 0.00391596305004663f}, {0.00107438975803964f, 0.995065715237115f, 0.00385989467364532f},
    {0.00106327904468655f, 0.99513049133726f, 0.00380622930570586f}, {0.00105254847301566f, 0.995192376992653f, 0.00375507423229327f},
    {0.00104226962376743f, 0.995251212297542f, 0.00370651779073142f}, {0.00103251630213132f, 0.995306839002218f, 0.00366064442174718f},
    {0.00102336575919332f, 0.99535908122064f, 0.00361755276006225f}, {0.0010148875099738f, 0.995407755600887f, 0.00357735664476775f},
    {0.00100714743087891f, 0.995452664088211f, 0.00354018824601941f}, {0.00100020772645907f, 0.995493605880918f, 0.00350618616349645f},
    {0.000994125554320192f, 0.995530387771965f, 0.00347548646268983f}, {0.000989926960592362f, 0.99554321506467f, 0.0034668577747009f},
    {0.000991948494702743f, 0.995497510348077f, 0.00351054096262953f}, {0.00098671008139981f, 0.995531526937004f, 0.00348176278783334f},
    {0.000987616131106955f, 0.99554321497737f, 0.00346916872609295f}, {0.000990739109245901f, 0.995520046419803f, 0.00348921429895365f},
    {0.00100095814751352f, 0.995423737690503f, 0.00357530400801253f}, {0.000997854413296231f, 0.995513566484496f, 0.00348857893771747f},
    {0.00100025563925044f, 0.995543214534084f, 0.00345652968956841f}, {0.00100582960564373f, 0.995537358480228f, 0.00345681178555054f},
    {0.00101103444907204f, 0.995543214885874f, 0.00344575053416357f}, {0.00101836470571876f, 0.995520638106343f, 0.00346099706992487f},
    {0.00102583697481177f, 0.995497471616308f, 0.00347669129489027f}, {0.00103360186414059f, 0.995473442631189f, 0.00349295540560879f},
    {0.0010418518712566f, 0.995447704444406f, 0.00351044358909034f}, {0.00105077832995165f, 0.995419417709551f, 0.00352980387571236f},
    {0.00106057016705192f, 0.995387766205211f, 0.00355166354883467f}, {0.00107140538599571f, 0.995352029022475f, 0.0035765655161336f},
    {0.00108346861367711f, 0.995311476680984f, 0.00360505463528331f}, {0.00109695575857428f, 0.995265341928867f, 0.00363770225523469f},
    {0.00111207603877874f, 0.995212813053082f, 0.00367511085173541f}, {0.00112899571787142f, 0.995153277871117f, 0.00371772635416002f},
    {0.00114766180235258f, 0.995087080512621f, 0.00376525764227259f}, {0.00116797249435672f, 0.995014784310207f, 0.0038172431573299f},
    {0.0011898337199143f, 0.994936929361246f, 0.00387323688317441f}, {0.00121316308520745f, 0.994854019433739f, 0.00393281745719606f},
    {0.00123798089476159f, 0.994766155185609f, 0.00399586391102516f}, {0.00126468492671321f, 0.994672035539955f, 0.00406327952871428f},
    {0.00129386318953173f, 0.994569245303462f, 0.0041368915146827f}, {0.00132607312406012f, 0.99445616654204f, 0.00421776035325079f},
    {0.00136199274322546f, 0.994330383860827f, 0.00430762342964648f}, {0.00140218827558767f, 0.994190097607497f, 0.00440771416843773f},
    {0.00144657027980234f, 0.994036044265193f, 0.00451738551819895f}, {0.0014948414463994f, 0.993869748034934f, 0.00463541058975732f},
    {0.00154664697157499f, 0.993692933166304f, 0.00476041994827885f}, {0.00160155611031673f, 0.99350757585286f, 0.00489086813480547f},
    {0.00165934351590016f, 0.99331488522358f, 0.0050257713662265f}, {0.00172098970224122f, 0.993111819487787f, 0.00516719092074947f},
    {0.00178799311734439f, 0.992893717152543f, 0.00531828984558922f}, {0.00186215733594893f, 0.992655160595531f, 0.0054826822156781f},
    {0.00194567722982058f, 0.992389796604603f, 0.00566452632394037f}, {0.00204086187457851f, 0.992091428392607f, 0.00586770989006352f},
    {0.00214886544504434f, 0.991758320093876f, 0.00609281463710752f}, {0.00227060126556373f, 0.991389991655376f, 0.00633940725496502f},
    {0.00240710194018797f, 0.990986049346417f, 0.00660684890039683f}, {0.00255951308567156f, 0.990546281797964f, 0.00689420534461247f},
    {0.00272961003173243f, 0.990069109916685f, 0.00720128027028609f}, {0.00292172449778344f, 0.98954595110701f, 0.00753232463362984f},
    {0.00314209112563573f, 0.988964336016287f, 0.00789357309271495f}, {0.00339901430740703f, 0.988308532045919f, 0.00829245388040288f},
    {0.0037036775131452f, 0.987559367177946f, 0.00873695559222785f}, {0.00407200203530361f, 0.986688050028538f, 0.00923994823615614f},
    {0.00452541495588699f, 0.985665328561487f, 0.00980925675224465f}, {0.00509650009960986f, 0.984445619212887f, 0.0104578809719572f},
    {0.00583624568025899f, 0.982963667419555f, 0.0112000872332953f}, {0.00682876239517714f, 0.981120811222337f, 0.0120504266988633f},
    {0.00822405389000236f, 0.978747717078863f, 0.0130282292907272f}, {0.0102850220872316f, 0.975586251520629f, 0.0141287269296979f},
    {0.013574328845884f, 0.971070041435139f, 0.0153556301518601f}, {0.0193066305447665f, 0.96414625633196f, 0.0165471134285413f},
    {0.031289649321998f, 0.951109697178212f, 0.0176006555471729f}, {0.0668370590231314f, 0.914977989973778f, 0.0181849523033105f},
    {0.47055821975157f, 0.511444995583411f, 0.0179967968958568f}, {0.910322527198126f, 0.0726668485313894f, 0.0170106249605477f},
    {0.948080455313271f, 0.0361923080770627f, 0.0157272385852104f}, {0.961821356060462f, 0.0237477481751048f, 0.0144308959859631f},
    {0.969199115942889f, 0.0175698872533994f, 0.0132309972432555f}, {0.973908662984669f, 0.0139226460381375f, 0.0121686912671749f},
    {0.977209973244265f, 0.0115405563250229f, 0.0112494709794463f}, {0.979656739520468f, 0.00987936539339772f, 0.0104638955435151f},
    {0.981533157171168f, 0.00866834659880336f, 0.00979849656878104f}, {0.98300459261219f, 0.00775698136505557f, 0.00923842640607187f},
    {0.984186034298509f, 0.0070497600039875f, 0.00876420604257239f}, {0.985154144725996f, 0.00648659344310688f, 0.00835926216828412f},
    {0.985960394124136f, 0.00602876059685385f, 0.00801084569654375f}, {0.986640780839053f, 0.00565019506385772f, 0.00770902447378607f},
    {0.987219960775549f, 0.00533339573118596f, 0.00744664385732223f}, {0.987712462684063f, 0.00506760323360109f, 0.00721993441762183f},
    {0.98812922337855f, 0.00484481984619499f, 0.00702595719156639f}, {0.988479466505093f, 0.00465858261733583f, 0.00686195128437309f},
    {0.988771011112271f, 0.00450363052254957f, 0.00672535880950215f}, {0.989010580213688f, 0.00437559807374124f, 0.00661382217947227f},
    {0.9892042685635f, 0.00427064636886897f, 0.00652508545411795f}, {0.989267381249336f, 0.00421399837222459f, 0.0065186207014573f},
    {0.98926740534975f, 0.0041821610608368f, 0.00655043395806412f}, {0.989234375147028f, 0.00415983508196995f, 0.00660579020001557f},
    {0.988677031131798f, 0.00429505524467431f, 0.00702791405273997f}, {0.989261896340495f, 0.00411521650646233f, 0.00662288754111876f},
    {0.989216667360449f, 0.00410703662046563f, 0.00667629645389074f}, {0.98908381615834f, 0.00413798119759596f, 0.00677820303410443f},
    {0.989159735413925f, 0.00410561424141076f, 0.00673465078490153f}, {0.989149438069669f, 0.00410210406035243f, 0.00674845823809849f},
    {0.989267414017139f, 0.00407205376923983f, 0.00666053260312171f}, {0.989267347297893f, 0.00406867675841742f, 0.0066639763143222f},
    {0.989149465414196f, 0.00410374302842169f, 0.00674679197440041f}, {0.989010741739061f, 0.00414689990776415f, 0.00684235874466237f},
    {0.988851556765261f, 0.00419805577549299f, 0.00695038786462158f}, {0.988666611701783f, 0.0042592412800962f, 0.00707414740744478f},
    {0.988449830921947f, 0.00433277711218761f, 0.00721739228664503f}, {0.988195045723736f, 0.00442099331156629f, 0.00738396127962477f},
    {0.987895790071809f, 0.00452631586678535f, 0.00757789432283689f}, {0.987547682812115f, 0.00465035300600109f, 0.00780196443975241f},
    {0.987156395994108f, 0.00479084335761819f, 0.00805276096208005f}, {0.986730687865098f, 0.00494437506976931f, 0.00832493739352415f},
    {0.986280206086251f, 0.0051072340743429f, 0.00861256011031588f}, {0.985815725823163f, 0.00527530812606157f, 0.00890896633417709f},
    {0.985346439308458f, 0.00544510677638383f, 0.00920845415055283f}, {0.984870899926437f, 0.00561716707756873f, 0.00951193322145867f},
    {0.984384353309599f, 0.0057932674062179f, 0.00982237946648336f}, {0.983881364727288f, 0.00597543537475818f, 0.0101432000722108f},
    {0.983355832537928f, 0.00616594246133098f, 0.0104782251474164f}, {0.982801681826561f, 0.00636703393642332f, 0.0108312843643075f},
    {0.982214944276448f, 0.0065801254183601f, 0.0112049304064262f}, {0.981591725416592f, 0.00680659697277952f, 0.0116016776831345f},
    {0.98092754126865f, 0.00704804585665008f, 0.0120244129683026f}, {0.980217090262774f, 0.00730637615403068f, 0.0124765336152861f},
    {0.979453267798003f, 0.00758420524152908f, 0.0129625269257405f}, {0.978624519621515f, 0.0078859723879099f, 0.0134895079454178f},
    {0.977717527383843f, 0.00821680986648768f, 0.01406566272648f}, {0.976717975770224f, 0.00858221875205089f, 0.0146998053590432f},
    {0.975610130314012f, 0.00898824321526453f, 0.0154016262328342f}, {0.974380955661223f, 0.00943989050166923f, 0.016179153503124f},
    {0.97303407501227f, 0.00993577225357135f, 0.0170301523769722f}, {0.971579295046055f, 0.0104721558632539f, 0.0179485487499031f},
    {0.970029137485527f, 0.0110442972001875f, 0.0189265649081134f}, {0.968399706730058f, 0.0116460953881084f, 0.0199541973030008f},
    {0.966705025849687f, 0.0122722369902735f, 0.021022736604104f}, {0.964937629361706f, 0.0129255280066701f, 0.0221368420177389f},
    {0.963082130442705f, 0.0136117759302372f, 0.0233060928289402f}, {0.961120546417721f, 0.0143377754175333f, 0.0245416773622041f},
    {0.9590319139876f, 0.0151114621734259f, 0.0258566227866761f}, {0.956796175514438f, 0.0159404740186431f, 0.027263349512981f},
    {0.954407851284952f, 0.016827045662572f, 0.0287651018742996f}, {0.951866676073972f, 0.0177714865078491f, 0.0303618362319376f},
    {0.949173892094391f, 0.0187735660724978f, 0.0320525403474783f}, {0.946332511974346f, 0.0198324130524218f, 0.0338350735352774f},
    {0.943345001728747f, 0.0209473786587602f, 0.0357076179263381f}, {0.940203648515703f, 0.0221216857161372f, 0.0376746637708886f},
    {0.936897409987339f, 0.0233598532330812f, 0.0397427346744222f}, {0.933414159950306f, 0.0246668596885005f, 0.0419189783494429f},
    {0.929740524243158f, 0.0260482124393599f, 0.0442112610380524f}, {0.925855987461467f, 0.027512146274089f, 0.0466318636851517f},
    {0.921711471921493f, 0.0290777114165444f, 0.0492108139101391f}, {0.917243538045007f, 0.0307695668950082f, 0.0519868921571845f},
    {0.912378106804082f, 0.032616739271856f, 0.0550051506612125f}, {0.907028952197872f, 0.0346533860677326f, 0.0583176583842104f},
    {0.901115003869107f, 0.0369124928322675f, 0.061972499429983f}, {0.894627709019822f, 0.0394004375408431f, 0.0659718493328085f},
    {0.887588778616097f, 0.0421129926205522f, 0.0702982242094708f}, {0.880035520058126f, 0.045040602520688f, 0.0749238723495664f},
    {0.872022572079822f, 0.0481669710569523f, 0.0798104513613239f}, {0.863596414980893f, 0.0514780312477339f, 0.0849255479810375f},
    {0.854704135853001f, 0.0549973448052305f, 0.0902985129470983f}, {0.845263865037901f, 0.0587615126664724f, 0.0959746151561844f},
    {0.835164234228834f, 0.0628159425352819f, 0.102019815549722f}, {0.824272187898797f, 0.0672139629653747f, 0.108513841066305f},
    {0.81243615811694f, 0.0720139577867945f, 0.115549875361241f}, {0.799554520195097f, 0.077255705344938f, 0.1231897643896f},
    {0.785585551400888f, 0.0829687581926738f, 0.131445679085806f}, {0.77059797235583f, 0.0891658605651413f, 0.140236154887926f},
    {0.754760688368665f, 0.0958388811531214f, 0.149400416994105f}, {0.738236975719522f, 0.102963396129037f, 0.158799613480885f},
    {0.721183518780123f, 0.110470201598867f, 0.168346263538835f}, {0.703725364167222f, 0.11828312573515f, 0.177991492572075f},
    {0.686168541918471f, 0.126264025632914f, 0.187567413274633f}, {0.6691014235548f, 0.134160995220305f, 0.196737560163804f},
    {0.653109491273661f, 0.141709675782503f, 0.205180810262307f}, {0.638215526478275f, 0.148883288878905f, 0.212901160346387f},
    {0.624181645705891f, 0.155766976057149f, 0.220051352305384f}, {0.610690212568645f, 0.162488299800432f, 0.226821459767806f},
    {0.597407438389556f, 0.169196694433434f, 0.233395837485818f}, {0.584083552723607f, 0.176016114608081f, 0.239900300561521f},
    {0.570748939335397f, 0.182939507363671f, 0.246311519008544f}, {0.557518389511387f, 0.189919996534704f, 0.252561577091982f},
    {0.544522552139376f, 0.196899404662466f, 0.258578003504021f}, {0.531898809916482f, 0.203808267712451f, 0.264292879711033f},
    {0.519754395007152f, 0.210577676294061f, 0.269667882798696f}, {0.50774930822063f, 0.21713871300938f, 0.275111929440529f},
    {0.49724655471009f, 0.223498311608695f, 0.279255080763256f}, {0.486917069519072f, 0.2295842389745f, 0.283498634632749f},
    {0.47720566645071f, 0.235392602481688f, 0.287401670206982f}, {0.468109700074472f, 0.240926940143867f, 0.290963294594973f},
    {0.459397412658735f, 0.246290143370965f, 0.294312374095256f}, {0.450946008343407f, 0.251568634559499f, 0.297485282122744f},
    {0.442614488798001f, 0.256839992043789f, 0.300545438238409f}, {0.434311955978638f, 0.262161160974956f, 0.303526795380919f},
    {0.426023875819186f, 0.267542374864477f, 0.306433653803779f}, {0.417917421815236f, 0.27287378975905f, 0.309208683825244f},
    {0.410169845563115f, 0.278031834508319f, 0.31179820459365f}, {0.402850283906071f, 0.282970590100404f, 0.314178998856179f},
    {0.396077350690941f, 0.287593834369213f, 0.316328674249106f}, {0.389957444122094f, 0.291811256524333f, 0.318231143498422f},
    {0.384417756099057f, 0.295673660400157f, 0.319908411332678f}, {0.37958956788575f, 0.299066827112916f, 0.321343415092933f},
    {0.375505813949727f, 0.301955363715723f, 0.322538614543891f}, {0.372054455444386f, 0.304416721465106f, 0.323528597323951f},
    {0.369258422307495f, 0.306422085971136f, 0.324319249151433f}, {0.366974936529127f, 0.308066808807225f, 0.324957996229284f},
    {0.365027410765379f, 0.309475417494957f, 0.325496897646785f}, {0.363270582875766f, 0.310752243609656f, 0.325976883780891f},
    {0.361611551321096f, 0.311963649081228f, 0.32642449345612f}, {0.360006491185019f, 0.313140149590384f, 0.326853035168552f},
    {0.358455027259584f, 0.314281405449835f, 0.327263223798044f}, {0.356957753201009f, 0.315386941585728f, 0.327654940204382f},
    {0.355516113074104f, 0.316455309958497f, 0.328028188565432f}, {0.354131597457431f, 0.317485010349718f, 0.328382978629746f},
    {0.352812618332759f, 0.318468978632809f, 0.328717961510659f}, {0.351570767660263f, 0.319398012508374f, 0.329030748002959f},
    {0.350406978974473f, 0.320271228184325f, 0.329321288886345f}, {0.349323662444366f, 0.321086111685236f, 0.329589688055097f},
    {0.3483254723968f, 0.321838652231916f, 0.329835302322305f}, {0.347409953666681f, 0.322530302812563f, 0.330059133618807f},
    {0.346551839316617f, 0.323179779309591f, 0.330267732556819f}, {0.345725463931662f, 0.323806292905732f, 0.330467551681723f},
    {0.344911242176026f, 0.324424632544639f, 0.330663385721125f}, {0.344094923178749f, 0.325045635561157f, 0.330858646184776f},
    {0.343272617544637f, 0.325672286215913f, 0.331054235605566f}, {0.34246463381076f, 0.326289100289123f, 0.331245329679247f},
    {0.341691657933856f, 0.326880178250347f, 0.331427141747785f}, {0.340978992446758f, 0.327423294574038f, 0.331596594724302f},
    {0.340308546669854f, 0.327940178493338f, 0.331750050921398f}, {0.339714434145185f, 0.328396403734122f, 0.331887825172607f},
    {0.339178020313029f, 0.328808778307809f, 0.332011741956555f}, {0.338688989226496f, 0.329185095822886f, 0.332124322764153f},
    {0.338238793819259f, 0.32953183146608f, 0.332227636907457f}, {0.337820387223979f, 0.329854341379531f, 0.332323372176796f},
    {0.337428358803245f, 0.33015673938345f, 0.33241282155364f}, {0.337059593960997f, 0.330441382144034f, 0.332496738385965f},
    {0.336711799297235f, 0.330710003016331f, 0.332575677698766f}, {0.336393489134285f, 0.330953530738308f, 0.33265018939778f},
    {0.336071726615592f, 0.331204759553083f, 0.332720407186174f}, {0.335778961551176f, 0.331431209900532f, 0.332786351701114f},
    {0.335514753467587f, 0.331635640205613f, 0.332845710523242f}, {0.335288428531529f, 0.331810798234289f, 0.332896429223342f},
    {0.335106400805813f, 0.331951691503971f, 0.332937121502569f}, {0.334982830237101f, 0.332045015184259f, 0.332966982226614f},
    {0.33488622718803f, 0.33212213939852f, 0.332986177196332f}, {0.334833532894243f, 0.332162955473576f, 0.332997867533508f},
    {0.334800885693187f, 0.332188262946562f, 0.333005085140322f}, {0.334777409959641f, 0.332206475043291f, 0.333010258369814f},
    {0.334754648816851f, 0.332224131172513f, 0.333015272788072f}, {0.334726747562834f, 0.3322457566078f, 0.3330214331626f},
    {0.334691498889229f, 0.332273056920239f, 0.333029228124995f}, {0.334647849086408f, 0.332306842183399f, 0.333038890806484f},
    {0.334595164551481f, 0.332347598898901f, 0.333050556332937f}, {0.334533170699357f, 0.332395532233052f, 0.333064278543963f},
    {0.334462673481989f, 0.332450009776818f, 0.333079868893349f}, {0.334387428202622f, 0.332508119588588f, 0.333096484008189f},
    {0.334321057590764f, 0.332557055569046f, 0.333113311257929f}, {0.334236779489025f, 0.332624321950136f, 0.333129634721204f},
    {0.334176374645139f, 0.332668565221611f, 0.333145037096858f}, {0.334102053304267f, 0.332728017392087f, 0.33315909121194f},
    {0.334044269237699f, 0.332772397765162f, 0.333171643842437f}, {0.334003485294713f, 0.332811116613815f, 0.333172848857627f},
    {0.333960612549575f, 0.332834225679941f, 0.33319177783655f}, {0.333915928070901f, 0.332870662692583f, 0.333199256746151f}};

  if ((spect.wavelength < spectrum::kRGBResponseShortestWavelength) || (spect.wavelength > spectrum::kRGBResponseLongestWavelength))
    return {spect, 0.0f};

  uint32_t wi = uint32_t(spect.wavelength - spectrum::kRGBResponseShortestWavelength);
  uint32_t wj = min(wi + 1u, spectrum::RGBResponseWavelengthCount - 1u);

  float dw = spect.wavelength - floorf(spect.wavelength);
  float3 w = lerp(response[wi], response[wj], dw);

  return {spect, rgb.x * w.x + rgb.y * w.y + rgb.z * w.z};
}

void init_spectrums(Spectrums&);

}  // namespace rgb

}  // namespace etx
